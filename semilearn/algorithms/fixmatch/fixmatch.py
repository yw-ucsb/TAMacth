# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import ConsistencyLossS, USHook, V4Hook, V5Hook, OBSHook, V6Hook

from torchmetrics.classification import MulticlassAccuracy

# Code to experiment V6;
@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):
    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")

        # OBS target hook;
        obs_target_dict = {
            'p_model': np.zeros(
                [self.args.num_train_iter // self.args.num_log_iter + 1,
                 self.args.num_classes]
            ),
            'conf_mat': np.zeros(
                [self.args.num_train_iter // self.args.num_eval_iter + 1,
                 self.args.num_classes,
                 self.args.num_classes]
            )
        }
        self.register_hook(OBSHook(obs_target_dict), "OBSHook")

        if self.args.dist_align_uw:
            self.register_hook(
                V6Hook(num_classes=self.num_classes, momentum=self.args.ema_p,
                                 p_target_type='uniform' if self.args.dist_uniform else 'model'),
                "UAWeightingHook")

        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # V6: do re-scaling and apply re-weighting on the 're-scaled' pseudo label
            probs_x_ulb_w = self.call_hook("dist_align", "UAWeightingHook", probs_x_ulb=probs_x_ulb_w.detach())

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            # Compute weighting per class;
            if self.registered_hook('UAWeightingHook'):
                weight_mask = self.call_hook("masking", "UAWeightingHook", probs_x_ulb=probs_x_ulb_w)
                mask = mask * weight_mask


            # generate unlabeled targets using pseudo label hook
            pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                                  pseudo_label_w,
                                                  'ce',
                                                  mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio_w=mask.float().mean().item())
        return out_dict, log_dict

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                # Load data and send to device;
                x = data["x_lb"]
                y = data["y_lb"]
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                # Calculate batch level metrics;
                logits = self.model(x)[out_key]
                loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)

                # Merge batch level metric into a list, or sum to be processed later after iterating full set;
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch

        # Calculate dataset level metrics;
        # Convert list into np arrays;
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        # Calculate dataset metrics here;
        # Default metrics from usb;
        top1 = accuracy_score(y_true, y_pred)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        F1 = f1_score(y_true, y_pred, average="macro")
        cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
        self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))

        self.call_hook("update_and_save_obs_target", "OBSHook", target_dict={'conf_mat': cf_mat})

        # Other metrics we want to monitor;
        top5 = MulticlassAccuracy(num_classes=self.args.num_classes, top_k=5)(torch.tensor(y_logits),
                                                                              torch.tensor(y_true)).item()

        if self.ema is not None:
            self.ema.restore()

        self.model.train()

        eval_dict = {
            eval_dest + "/loss": total_loss / total_num,
            eval_dest + "/top-1-acc": top1,
            eval_dest + "/balanced_acc": balanced_top1,
            eval_dest + "/precision": precision,
            eval_dest + "/recall": recall,
            eval_dest + "/F1": F1,
            eval_dest + "/top-5-acc": top5,
        }
        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_logits
        return eval_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]


# Old code for obs v3 and none;
# class FixMatch(AlgorithmBase):
#     """
#         FixMatch algorithm (https://arxiv.org/abs/2001.07685).
#
#         Args:
#             - args (`argparse`):
#                 algorithm arguments
#             - net_builder (`callable`):
#                 network loading function
#             - tb_log (`TBLog`):
#                 tensorboard logger
#             - logger (`logging.Logger`):
#                 logger to use
#             - T (`float`):
#                 Temperature for pseudo-label sharpening
#             - p_cutoff(`float`):
#                 Confidence threshold for generating pseudo-labels
#             - hard_label (`bool`, *optional*, default to `False`):
#                 If True, targets have [Batch size] shape with int values. If False, the target is vector
#     """
#
#     def __init__(self, args, net_builder, tb_log=None, logger=None):
#         super().__init__(args, net_builder, tb_log, logger)
#         # fixmatch specified arguments
#         self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
#
#     def init(self, T, p_cutoff, hard_label=True):
#         self.T = T
#         self.p_cutoff = p_cutoff
#         self.use_hard_label = hard_label
#
#     def set_hooks(self):
#         self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
#         self.register_hook(FixedThresholdingHook(), "MaskingHook")
#
#         # OBS target hook;
#         obs_target_dict = {
#             'p_model': np.zeros(
#                 [self.args.num_train_iter // self.args.num_log_iter + 1,
#                 self.args.num_classes]
#             ),
#             'conf_mat': np.zeros(
#                 [self.args.num_train_iter // self.args.num_eval_iter + 1,
#                 self.args.num_classes,
#                 self.args.num_classes]
#             )
#         }
#         self.register_hook(OBSHook(obs_target_dict), "OBSHook")
#
#         # UA hook;
#         self.register_hook(
#             V5Hook(num_classes=self.num_classes, momentum=self.args.ema_p,
#                              p_target_type='uniform' if self.args.dist_uniform else 'model'),
#             "UAWeightingHook")
#
#         super().set_hooks()
#
#     def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
#         num_lb = y_lb.shape[0]
#
#         # inference and calculate sup/unsup losses
#         with self.amp_cm():
#             if self.use_cat:
#                 inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
#                 outputs = self.model(inputs)
#                 logits_x_lb = outputs['logits'][:num_lb]
#                 logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
#                 feats_x_lb = outputs['feat'][:num_lb]
#                 feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
#             else:
#                 outs_x_lb = self.model(x_lb)
#                 logits_x_lb = outs_x_lb['logits']
#                 feats_x_lb = outs_x_lb['feat']
#                 outs_x_ulb_s = self.model(x_ulb_s)
#                 logits_x_ulb_s = outs_x_ulb_s['logits']
#                 feats_x_ulb_s = outs_x_ulb_s['feat']
#                 with torch.no_grad():
#                     outs_x_ulb_w = self.model(x_ulb_w)
#                     logits_x_ulb_w = outs_x_ulb_w['logits']
#                     feats_x_ulb_w = outs_x_ulb_w['feat']
#             feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}
#
#             sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
#
#             # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
#             probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
#
#             # if distribution alignment hook is registered, call it
#             # this is implemented for imbalanced algorithm - CReST
#             if self.args.dist_align_uw:
#                 probs_x_ulb_w_ = self.call_hook("dist_align", "UAWeightingHook", probs_x_ulb=probs_x_ulb_w.detach())
#
#             # Generating binary mask based on ua-aligned scaled smax, but do weighting based on the class prediction;
#             mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
#
#             # Compute weighting per class;
#             # if self.registered_hook('UAWeightingHook'):
#
#             # weight_mask = self.call_hook("masking", "UAWeightingHook", probs_x_ulb=probs_x_ulb_w)
#             # mask = mask * weight_mask
#
#             # generate unlabeled targets using pseudo label hook;
#             pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
#                                           logits=probs_x_ulb_w,
#                                           use_hard_label=self.use_hard_label,
#                                           T=self.T,
#                                           softmax=False)
#
#             unsup_loss = self.consistency_loss(logits_x_ulb_s,
#                                                   pseudo_label_w,
#                                                   'ce',
#                                                   mask=mask)
#
#             total_loss = sup_loss + self.lambda_u * unsup_loss
#
#         out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
#         log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
#                                          unsup_loss=unsup_loss.item(),
#                                          total_loss=total_loss.item(),
#                                          util_ratio_w=mask.float().mean().item())
#         return out_dict, log_dict
#
#     def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
#         """
#         evaluation function
#         """
#         self.model.eval()
#         if self.ema is not None:
#             self.ema.apply_shadow()
#
#         eval_loader = self.loader_dict[eval_dest]
#         total_loss = 0.0
#         total_num = 0.0
#         y_true = []
#         y_pred = []
#         # y_probs = []
#         y_logits = []
#         with torch.no_grad():
#             for data in eval_loader:
#                 # Load data and send to device;
#                 x = data["x_lb"]
#                 y = data["y_lb"]
#                 if isinstance(x, dict):
#                     x = {k: v.cuda(self.gpu) for k, v in x.items()}
#                 else:
#                     x = x.cuda(self.gpu)
#                 y = y.cuda(self.gpu)
#
#                 num_batch = y.shape[0]
#                 total_num += num_batch
#
#                 # Calculate batch level metrics;
#                 logits = self.model(x)[out_key]
#                 loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
#
#                 # Merge batch level metric into a list, or sum to be processed later after iterating full set;
#                 y_true.extend(y.cpu().tolist())
#                 y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
#                 y_logits.append(logits.cpu().numpy())
#                 total_loss += loss.item() * num_batch
#
#         # Calculate dataset level metrics;
#         # Convert list into np arrays;
#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)
#         y_logits = np.concatenate(y_logits)
#
#         # Calculate dataset metrics here;
#         # Default metrics from usb;
#         top1 = accuracy_score(y_true, y_pred)
#         balanced_top1 = balanced_accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="macro")
#         recall = recall_score(y_true, y_pred, average="macro")
#         F1 = f1_score(y_true, y_pred, average="macro")
#         cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
#         self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))
#
#         self.call_hook("update_and_save_obs_target", "OBSHook", target_dict={'conf_mat': cf_mat})
#
#         # Other metrics we want to monitor;
#         top5 = MulticlassAccuracy(num_classes=self.args.num_classes, top_k=5)(torch.tensor(y_logits),
#                                                                               torch.tensor(y_true)).item()
#
#         if self.ema is not None:
#             self.ema.restore()
#
#         self.model.train()
#
#         eval_dict = {
#             eval_dest + "/loss": total_loss / total_num,
#             eval_dest + "/top-1-acc": top1,
#             eval_dest + "/balanced_acc": balanced_top1,
#             eval_dest + "/precision": precision,
#             eval_dest + "/recall": recall,
#             eval_dest + "/F1": F1,
#             eval_dest + "/top-5-acc": top5,
#         }
#         if return_logits:
#             eval_dict[eval_dest + "/logits"] = y_logits
#         return eval_dict
#
#     @staticmethod
#     def get_argument():
#         return [
#             SSL_Argument('--hard_label', str2bool, True),
#             SSL_Argument('--T', float, 0.5),
#             SSL_Argument('--p_cutoff', float, 0.95),
#         ]


# Old code for v4 that applies weighting to unscaled smax, based on scaled smax;

# @ALGORITHMS.register('fixmatch')
# class FixMatch(AlgorithmBase):
#     """
#         FixMatch algorithm (https://arxiv.org/abs/2001.07685).
#
#         Args:
#             - args (`argparse`):
#                 algorithm arguments
#             - net_builder (`callable`):
#                 network loading function
#             - tb_log (`TBLog`):
#                 tensorboard logger
#             - logger (`logging.Logger`):
#                 logger to use
#             - T (`float`):
#                 Temperature for pseudo-label sharpening
#             - p_cutoff(`float`):
#                 Confidence threshold for generating pseudo-labels
#             - hard_label (`bool`, *optional*, default to `False`):
#                 If True, targets have [Batch size] shape with int values. If False, the target is vector
#     """
#
#     def __init__(self, args, net_builder, tb_log=None, logger=None):
#         super().__init__(args, net_builder, tb_log, logger)
#         # fixmatch specified arguments
#         self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
#
#     def init(self, T, p_cutoff, hard_label=True):
#         self.T = T
#         self.p_cutoff = p_cutoff
#         self.use_hard_label = hard_label
#
#     def set_hooks(self):
#         self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
#         self.register_hook(FixedThresholdingHook(), "MaskingHook")
#
#         # OBS target hook;
#         obs_target_dict = {
#             'p_model': np.zeros(
#                 [self.args.num_train_iter // self.args.num_log_iter + 1,
#                  self.args.num_classes]
#             ),
#             'conf_mat': np.zeros(
#                 [self.args.num_train_iter // self.args.num_eval_iter + 1,
#                  self.args.num_classes,
#                  self.args.num_classes]
#             )
#         }
#         self.register_hook(OBSHook(obs_target_dict), "OBSHook")
#
#         if self.args.dist_align_uw:
#             self.register_hook(
#                 V4Hook(num_classes=self.num_classes, momentum=self.args.ema_p,
#                                  p_target_type='uniform' if self.args.dist_uniform else 'model'),
#                 "UAWeightingHook")
#
#         super().set_hooks()
#
#     def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
#         num_lb = y_lb.shape[0]
#
#         # inference and calculate sup/unsup losses
#         with self.amp_cm():
#             if self.use_cat:
#                 inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
#                 outputs = self.model(inputs)
#                 logits_x_lb = outputs['logits'][:num_lb]
#                 logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
#                 feats_x_lb = outputs['feat'][:num_lb]
#                 feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
#             else:
#                 outs_x_lb = self.model(x_lb)
#                 logits_x_lb = outs_x_lb['logits']
#                 feats_x_lb = outs_x_lb['feat']
#                 outs_x_ulb_s = self.model(x_ulb_s)
#                 logits_x_ulb_s = outs_x_ulb_s['logits']
#                 feats_x_ulb_s = outs_x_ulb_s['feat']
#                 with torch.no_grad():
#                     outs_x_ulb_w = self.model(x_ulb_w)
#                     logits_x_ulb_w = outs_x_ulb_w['logits']
#                     feats_x_ulb_w = outs_x_ulb_w['feat']
#             feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}
#
#             sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
#
#             # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
#             probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
#
#             # if distribution alignment hook is registered, call it
#             # this is implemented for imbalanced algorithm - CReST
#
#
#             # Generating the pl based on un scaled smax and but do weighting based on the class prediction;
#             mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
#             # Compute weighting per class;
#             if self.registered_hook('UAWeightingHook'):
#                 weight_mask = self.call_hook("masking", "UAWeightingHook", probs_x_ulb=probs_x_ulb_w)
#                 mask = mask * weight_mask
#
#
#             # generate unlabeled targets using pseudo label hook
#             pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
#                                           logits=probs_x_ulb_w,
#                                           use_hard_label=self.use_hard_label,
#                                           T=self.T,
#                                           softmax=False)
#
#             unsup_loss = self.consistency_loss(logits_x_ulb_s,
#                                                   pseudo_label_w,
#                                                   'ce',
#                                                   mask=mask)
#
#             total_loss = sup_loss + self.lambda_u * unsup_loss
#
#         out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
#         log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
#                                          unsup_loss=unsup_loss.item(),
#                                          total_loss=total_loss.item(),
#                                          util_ratio_w=mask.float().mean().item())
#         return out_dict, log_dict
#
#     def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
#         """
#         evaluation function
#         """
#         self.model.eval()
#         if self.ema is not None:
#             self.ema.apply_shadow()
#
#         eval_loader = self.loader_dict[eval_dest]
#         total_loss = 0.0
#         total_num = 0.0
#         y_true = []
#         y_pred = []
#         # y_probs = []
#         y_logits = []
#         with torch.no_grad():
#             for data in eval_loader:
#                 # Load data and send to device;
#                 x = data["x_lb"]
#                 y = data["y_lb"]
#                 if isinstance(x, dict):
#                     x = {k: v.cuda(self.gpu) for k, v in x.items()}
#                 else:
#                     x = x.cuda(self.gpu)
#                 y = y.cuda(self.gpu)
#
#                 num_batch = y.shape[0]
#                 total_num += num_batch
#
#                 # Calculate batch level metrics;
#                 logits = self.model(x)[out_key]
#                 loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
#
#                 # Merge batch level metric into a list, or sum to be processed later after iterating full set;
#                 y_true.extend(y.cpu().tolist())
#                 y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
#                 y_logits.append(logits.cpu().numpy())
#                 total_loss += loss.item() * num_batch
#
#         # Calculate dataset level metrics;
#         # Convert list into np arrays;
#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)
#         y_logits = np.concatenate(y_logits)
#
#         # Calculate dataset metrics here;
#         # Default metrics from usb;
#         top1 = accuracy_score(y_true, y_pred)
#         balanced_top1 = balanced_accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="macro")
#         recall = recall_score(y_true, y_pred, average="macro")
#         F1 = f1_score(y_true, y_pred, average="macro")
#         cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
#         self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))
#
#         self.call_hook("update_and_save_obs_target", "OBSHook", target_dict={'conf_mat': cf_mat})
#
#         # Other metrics we want to monitor;
#         top5 = MulticlassAccuracy(num_classes=self.args.num_classes, top_k=5)(torch.tensor(y_logits),
#                                                                               torch.tensor(y_true)).item()
#
#         if self.ema is not None:
#             self.ema.restore()
#
#         self.model.train()
#
#         eval_dict = {
#             eval_dest + "/loss": total_loss / total_num,
#             eval_dest + "/top-1-acc": top1,
#             eval_dest + "/balanced_acc": balanced_top1,
#             eval_dest + "/precision": precision,
#             eval_dest + "/recall": recall,
#             eval_dest + "/F1": F1,
#             eval_dest + "/top-5-acc": top5,
#         }
#         if return_logits:
#             eval_dict[eval_dest + "/logits"] = y_logits
#         return eval_dict
#
#     @staticmethod
#     def get_argument():
#         return [
#             SSL_Argument('--hard_label', str2bool, True),
#             SSL_Argument('--T', float, 0.5),
#             SSL_Argument('--p_cutoff', float, 0.95),
#         ]



# Old code for using s_s to predict pseudo label; -> not working;
# class FixMatch(AlgorithmBase):
#     """
#         FixMatch algorithm (https://arxiv.org/abs/2001.07685).
#
#         Args:
#             - args (`argparse`):
#                 algorithm arguments
#             - net_builder (`callable`):
#                 network loading function
#             - tb_log (`TBLog`):
#                 tensorboard logger
#             - logger (`logging.Logger`):
#                 logger to use
#             - T (`float`):
#                 Temperature for pseudo-label sharpening
#             - p_cutoff(`float`):
#                 Confidence threshold for generating pseudo-labels
#             - hard_label (`bool`, *optional*, default to `False`):
#                 If True, targets have [Batch size] shape with int values. If False, the target is vector
#     """
#
#     def __init__(self, args, net_builder, tb_log=None, logger=None):
#         super().__init__(args, net_builder, tb_log, logger)
#         # fixmatch specified arguments
#         self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
#
#     def init(self, T, p_cutoff, hard_label=True):
#         self.T = T
#         self.p_cutoff = p_cutoff
#         self.use_hard_label = hard_label
#
#     def set_hooks(self):
#         self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
#         self.register_hook(FixedThresholdingHook(), "MaskingHook")
#
#         if self.args.dist_align_uw:
#             self.register_hook(
#                 DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p,
#                                  p_target_type='uniform' if self.args.dist_uniform else 'model'),
#                 "DistAlignHookUW")
#         if self.args.dist_align_us:
#             self.register_hook(
#                 DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p,
#                                  p_target_type='uniform' if self.args.dist_uniform else 'model'),
#                 "DistAlignHookUS")
#
#         super().set_hooks()
#
#     def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
#         num_lb = y_lb.shape[0]
#
#         # inference and calculate sup/unsup losses
#         with self.amp_cm():
#             if self.use_cat:
#                 inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
#                 outputs = self.model(inputs)
#                 logits_x_lb = outputs['logits'][:num_lb]
#                 logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
#                 feats_x_lb = outputs['feat'][:num_lb]
#                 feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
#             else:
#                 outs_x_lb = self.model(x_lb)
#                 logits_x_lb = outs_x_lb['logits']
#                 feats_x_lb = outs_x_lb['feat']
#                 outs_x_ulb_s = self.model(x_ulb_s)
#                 logits_x_ulb_s = outs_x_ulb_s['logits']
#                 feats_x_ulb_s = outs_x_ulb_s['feat']
#                 with torch.no_grad():
#                     outs_x_ulb_w = self.model(x_ulb_w)
#                     logits_x_ulb_w = outs_x_ulb_w['logits']
#                     feats_x_ulb_w = outs_x_ulb_w['feat']
#             feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}
#
#             sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
#
#             # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
#             probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
#             probs_x_ulb_s = self.compute_prob(logits_x_ulb_s.detach())
#
#             # if distribution alignment hook is registered, call it
#             # this is implemented for imbalanced algorithm - CReST
#
#             unsup_loss = torch.tensor([0.], device=probs_x_ulb_w.device)
#             mask_w = torch.zeros(logits_x_ulb_w.shape[0], device=probs_x_ulb_w.device)
#             mask_s = torch.zeros(logits_x_ulb_s.shape[0], device=probs_x_ulb_w.device)
#
#             if self.args.uw:
#                 # Standard practice in UDA and Fixmatch: pl_w -> logits_s;
#                 if self.registered_hook("DistAlignHookUW"):
#                     probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHookUW", probs_x_ulb=probs_x_ulb_w.detach())
#                 # compute mask
#                 mask_w = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
#                 # generate unlabeled targets using pseudo label hook
#                 pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
#                                               logits=probs_x_ulb_w,
#                                               use_hard_label=self.use_hard_label,
#                                               T=self.T,
#                                               softmax=False)
#                 unsup_loss_uw = self.consistency_loss(logits_x_ulb_s,
#                                                       pseudo_label_w,
#                                                       'ce',
#                                                       mask=mask_w)
#                 unsup_loss += unsup_loss_uw
#
#             # New term
#             if self.args.us:
#                 if self.registered_hook("DistAlignHookUS"):
#                     probs_x_ulb_s = self.call_hook("dist_align", "DistAlignHookUS", probs_x_ulb=probs_x_ulb_s.detach())
#                 # compute mask
#                 mask_s = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_s, softmax_x_ulb=False)
#                 # generate unlabeled targets using pseudo label hook
#                 pseudo_label_s = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
#                                                 logits=probs_x_ulb_s,
#                                                 use_hard_label=self.use_hard_label,
#                                                 T=self.T,
#                                                 softmax=False)
#                 unsup_loss_us = self.consistency_loss(logits_x_ulb_w,
#                                                       pseudo_label_s,
#                                                       'ce',
#                                                       mask=mask_s)
#                 unsup_loss += unsup_loss_us
#
#             total_loss = sup_loss + self.lambda_u * unsup_loss
#
#         out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
#         log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
#                                          unsup_loss=unsup_loss.item(),
#                                          total_loss=total_loss.item(),
#                                          util_ratio_w=mask_w.float().mean().item(),
#                                          util_ratio_s=mask_s.float().mean().item())
#         return out_dict, log_dict
#
#     def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
#         """
#         evaluation function
#         """
#         self.model.eval()
#         if self.ema is not None:
#             self.ema.apply_shadow()
#
#         eval_loader = self.loader_dict[eval_dest]
#         total_loss = 0.0
#         total_num = 0.0
#         y_true = []
#         y_pred = []
#         # y_probs = []
#         y_logits = []
#         with torch.no_grad():
#             for data in eval_loader:
#                 # Load data and send to device;
#                 x = data["x_lb"]
#                 y = data["y_lb"]
#                 if isinstance(x, dict):
#                     x = {k: v.cuda(self.gpu) for k, v in x.items()}
#                 else:
#                     x = x.cuda(self.gpu)
#                 y = y.cuda(self.gpu)
#
#                 num_batch = y.shape[0]
#                 total_num += num_batch
#
#                 # Calculate batch level metrics;
#                 logits = self.model(x)[out_key]
#                 loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
#
#                 # Merge batch level metric into a list, or sum to be processed later after iterating full set;
#                 y_true.extend(y.cpu().tolist())
#                 y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
#                 y_logits.append(logits.cpu().numpy())
#                 total_loss += loss.item() * num_batch
#
#         # Calculate dataset level metrics;
#         # Convert list into np arrays;
#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)
#         y_logits = np.concatenate(y_logits)
#
#         # Calculate dataset metrics here;
#         # Default metrics from usb;
#         top1 = accuracy_score(y_true, y_pred)
#         balanced_top1 = balanced_accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="macro")
#         recall = recall_score(y_true, y_pred, average="macro")
#         F1 = f1_score(y_true, y_pred, average="macro")
#         cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
#         self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))
#
#         # Other metrics we want to monitor;
#         top5 = MulticlassAccuracy(num_classes=self.args.num_classes, top_k=5)(torch.tensor(y_logits),
#                                                                               torch.tensor(y_true)).item()
#
#         if self.ema is not None:
#             self.ema.restore()
#
#         self.model.train()
#
#         eval_dict = {
#             eval_dest + "/loss": total_loss / total_num,
#             eval_dest + "/top-1-acc": top1,
#             eval_dest + "/balanced_acc": balanced_top1,
#             eval_dest + "/precision": precision,
#             eval_dest + "/recall": recall,
#             eval_dest + "/F1": F1,
#             eval_dest + "/top-5-acc": top5,
#         }
#         if return_logits:
#             eval_dict[eval_dest + "/logits"] = y_logits
#         return eval_dict
#
#     @staticmethod
#     def get_argument():
#         return [
#             SSL_Argument('--hard_label', str2bool, True),
#             SSL_Argument('--T', float, 0.5),
#             SSL_Argument('--p_cutoff', float, 0.95),
#         ]

# Old files for testing applying ua to the smax_s (retain gradient);
# class FixMatch(AlgorithmBase):
#
#     """
#         FixMatch algorithm (https://arxiv.org/abs/2001.07685).
#
#         Args:
#             - args (`argparse`):
#                 algorithm arguments
#             - net_builder (`callable`):
#                 network loading function
#             - tb_log (`TBLog`):
#                 tensorboard logger
#             - logger (`logging.Logger`):
#                 logger to use
#             - T (`float`):
#                 Temperature for pseudo-label sharpening
#             - p_cutoff(`float`):
#                 Confidence threshold for generating pseudo-labels
#             - hard_label (`bool`, *optional*, default to `False`):
#                 If True, targets have [Batch size] shape with int values. If False, the target is vector
#     """
#     def __init__(self, args, net_builder, tb_log=None, logger=None):
#         super().__init__(args, net_builder, tb_log, logger)
#         # fixmatch specified arguments
#         self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
#
#         self.consistency_loss = ConsistencyLossS()
#
#     def init(self, T, p_cutoff, hard_label=True):
#         self.T = T
#         self.p_cutoff = p_cutoff
#         self.use_hard_label = hard_label
#
#     def set_hooks(self):
#         self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
#         self.register_hook(FixedThresholdingHook(), "MaskingHook")
#
#         if self.args.dist_align_uw:
#             self.register_hook(
#                 DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p,
#                                  p_target_type='uniform' if self.args.dist_uniform else 'model'),
#                 "DistAlignHookUW")
#         if self.args.dist_align_us:
#             self.register_hook(
#                 USHook(num_classes=self.num_classes, momentum=self.args.ema_p,
#                                  p_target_type='uniform' if self.args.dist_uniform else 'model'),
#                 "DistAlignHookUS")
#
#         super().set_hooks()
#
#     def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
#         num_lb = y_lb.shape[0]
#
#         # inference and calculate sup/unsup losses
#         with self.amp_cm():
#             if self.use_cat:
#                 inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
#                 outputs = self.model(inputs)
#                 logits_x_lb = outputs['logits'][:num_lb]
#                 logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
#                 feats_x_lb = outputs['feat'][:num_lb]
#                 feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
#             else:
#                 outs_x_lb = self.model(x_lb)
#                 logits_x_lb = outs_x_lb['logits']
#                 feats_x_lb = outs_x_lb['feat']
#                 outs_x_ulb_s = self.model(x_ulb_s)
#                 logits_x_ulb_s = outs_x_ulb_s['logits']
#                 feats_x_ulb_s = outs_x_ulb_s['feat']
#                 with torch.no_grad():
#                     outs_x_ulb_w = self.model(x_ulb_w)
#                     logits_x_ulb_w = outs_x_ulb_w['logits']
#                     feats_x_ulb_w = outs_x_ulb_w['feat']
#             feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
#
#             sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
#
#             # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
#             probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
#
#             # if distribution alignment hook is registered, call it
#             # this is implemented for imbalanced algorithm - CReST
#             if self.registered_hook("DistAlignHookUW"):
#                 probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHookUW", probs_x_ulb=probs_x_ulb_w.detach())
#
#             # compute mask
#             mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
#
#             # generate unlabeled targets using pseudo label hook
#             pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
#                                           logits=probs_x_ulb_w,
#                                           use_hard_label=self.use_hard_label,
#                                           T=self.T,
#                                           softmax=False)
#
#             if self.registered_hook("DistAlignHookUS"):
#                 probs_x_ulb_s = self.compute_prob(logits_x_ulb_s)
#                 probs_x_ulb_s = self.call_hook("dist_align", "DistAlignHookUS", probs_x_ulb=probs_x_ulb_s)
#                 unsup_loss = self.consistency_loss(probs_x_ulb_s,
#                                                    pseudo_label,
#                                                    'ce',
#                                                    mask=mask,
#                                                    softmax=True)
#             else:
#                 unsup_loss = self.consistency_loss(logits_x_ulb_s,
#                                                    pseudo_label,
#                                                    'ce',
#                                                    mask=mask,
#                                                    softmax=False)
#
#             total_loss = sup_loss + self.lambda_u * unsup_loss
#
#         out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
#         log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
#                                          unsup_loss=unsup_loss.item(),
#                                          total_loss=total_loss.item(),
#                                          util_ratio=mask.float().mean().item())
#         return out_dict, log_dict
#
#     def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
#         """
#         evaluation function
#         """
#         self.model.eval()
#         if self.ema is not None:
#             self.ema.apply_shadow()
#
#         eval_loader = self.loader_dict[eval_dest]
#         total_loss = 0.0
#         total_num = 0.0
#         y_true = []
#         y_pred = []
#         # y_probs = []
#         y_logits = []
#         with torch.no_grad():
#             for data in eval_loader:
#                 # Load data and send to device;
#                 x = data["x_lb"]
#                 y = data["y_lb"]
#                 if isinstance(x, dict):
#                     x = {k: v.cuda(self.gpu) for k, v in x.items()}
#                 else:
#                     x = x.cuda(self.gpu)
#                 y = y.cuda(self.gpu)
#
#                 num_batch = y.shape[0]
#                 total_num += num_batch
#
#                 # Calculate batch level metrics;
#                 logits = self.model(x)[out_key]
#                 loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
#
#                 # Merge batch level metric into a list, or sum to be processed later after iterating full set;
#                 y_true.extend(y.cpu().tolist())
#                 y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
#                 y_logits.append(logits.cpu().numpy())
#                 total_loss += loss.item() * num_batch
#
#         # Calculate dataset level metrics;
#         # Convert list into np arrays;
#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)
#         y_logits = np.concatenate(y_logits)
#
#         # Calculate dataset metrics here;
#         # Default metrics from usb;
#         top1 = accuracy_score(y_true, y_pred)
#         balanced_top1 = balanced_accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="macro")
#         recall = recall_score(y_true, y_pred, average="macro")
#         F1 = f1_score(y_true, y_pred, average="macro")
#         cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
#         self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))
#
#         # Other metrics we want to monitor;
#         top5 = MulticlassAccuracy(num_classes=self.args.num_classes, top_k=5)(torch.tensor(y_logits), torch.tensor(y_true)).item()
#
#         if self.ema is not None:
#             self.ema.restore()
#
#         self.model.train()
#
#         eval_dict = {
#             eval_dest + "/loss": total_loss / total_num,
#             eval_dest + "/top-1-acc": top1,
#             eval_dest + "/balanced_acc": balanced_top1,
#             eval_dest + "/precision": precision,
#             eval_dest + "/recall": recall,
#             eval_dest + "/F1": F1,
#             eval_dest + "/top-5-acc": top5,
#         }
#         if return_logits:
#             eval_dict[eval_dest + "/logits"] = y_logits
#         return eval_dict
#
#     @staticmethod
#     def get_argument():
#         return [
#             SSL_Argument('--hard_label', str2bool, True),
#             SSL_Argument('--T', float, 0.5),
#             SSL_Argument('--p_cutoff', float, 0.95),
#         ]
