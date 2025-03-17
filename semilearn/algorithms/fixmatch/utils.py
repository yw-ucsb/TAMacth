import torch
import torch.nn as nn

import pickle

import numpy as np
from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import concat_all_gather
from torch.nn import functional as F


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits, targets, name='ce', mask=None, softmax=False):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    if softmax == False:
        # logits_w = logits_w.detach()
        if name == 'mse':
            probs = torch.softmax(logits, dim=-1)
            loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
        else:
            loss = ce_loss(logits, targets, reduction='none')

    else:
        loss = F.nll_loss(torch.log(logits), targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()


class ConsistencyLossS(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None, softmax=False):
        return consistency_loss(logits, targets, name, mask, softmax)


class USHook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment;
    """

    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.update_p_target:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        # p_model: expectation of p, initially set to None, EMA update during training;
        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        # Softmatch: target type: uniform, update_p_target: False;
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target


class V4Hook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment;
    """

    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def masking(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)
        # Calculate the per-class weighting;
        weight_pre_class = (self.p_target + 1e-6) / (self.p_model + 1e-6)

        # Get the predicted class of unlabeled data;
        plb = torch.argmax(probs_x_ulb, dim=-1)

        mask = weight_pre_class[plb]
        return mask

    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.update_p_target:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        # p_model: expectation of p, initially set to None, EMA update during training;
        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        # Qmatch does not use prob_x_lb;
        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        # Softmatch: target type: uniform, update_p_target: False;
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target


# V5 Hook, masking will not update the p_model again;
class V5Hook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment;
    """

    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    @torch.no_grad()
    def masking(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        # self.update_p(algorithm, probs_x_ulb, probs_x_lb)
        # Calculate the per-class weighting;
        weight_pre_class = (self.p_target + 1e-6) / (self.p_model + 1e-6)

        # Get the predicted class of unlabeled data;
        plb = torch.argmax(probs_x_ulb, dim=-1)

        mask = weight_pre_class[plb]
        return mask

    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.update_p_target:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        # p_model: expectation of p, initially set to None, EMA update during training;
        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        # Qmatch does not use prob_x_lb;
        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        # Softmatch: target type: uniform, update_p_target: False;
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target


class V6Hook(Hook):
    """
    V6 Hook:
    1. Update of the p_model will be based on the 'raw' pseudo labels. 'Raw' means prediction with unscaled smax;
    2. Set bound for the scaling factor which is: [1/(1 + DKL[[p_model||Uniform]), 1 + DKL[p_model||Uniform]];
    Procedure:
        1. Given smax_u for the current batch, predict raw pseudo labels;
        2. Update the p_model with the raw pseudo labels;
        3. Calculate the scaling factor and apply the bound;
        4. Rescale and normalize smax_u with the bounded scaling factor;
        5. Generate the weighting mask with the bounded scaling factor and scaled smax_u;
        6. On the outside, the actual pseudo label will be predicted with the rescaled smax_u and be applied weighting;
    """

    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # First need to check
        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)

        scaler = self.get_bounded_scaler()
        probs_x_ulb_aligned = probs_x_ulb * scaler
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

        # # dist align
        # probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        # probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        # return probs_x_ulb_aligned

    @torch.no_grad()
    def masking(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        # self.update_p(algorithm, probs_x_ulb, probs_x_lb)
        # Calculate the per-class weighting;
        weight_pre_class = self.get_bounded_scaler()
        # weight_pre_class = (self.p_target + 1e-6) / (self.p_model + 1e-6)

        # Get the predicted class of unlabeled data;
        plb = torch.argmax(probs_x_ulb, dim=-1)

        mask = weight_pre_class[plb]
        return mask

    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        if algorithm.distributed and algorithm.world_size > 1:
            if probs_x_lb is not None and self.update_p_target:
                probs_x_lb = concat_all_gather(probs_x_lb)
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        # Calculate the raw pseudo label;
        probs_x_ulb = probs_x_ulb.detach()
        n_classes = probs_x_ulb.shape[1]

        # First check if there are any
        idx = self.find_indices_above_th(probs_x_ulb)
        if idx is not None:
            plb_one_hot = 1. * torch.nn.functional.one_hot(idx, num_classes=n_classes)
            p_plb_current = torch.mean(plb_one_hot, dim=0)

            if self.p_model is None:
                self.p_model = p_plb_current
            else:
                self.p_model = self.p_model * self.m + p_plb_current * (1 - self.m)

        # Qmatch does not use prob_x_lb;
        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)

    def set_p_target(self, p_target_type='uniform', p_target=None):
        # Softmatch: target type: uniform, update_p_target: False;
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target

    def find_indices_above_th(self, p_input, threshold=0.95):
        # Check if any element is greater than the threshold
        mask = p_input > threshold
        if not mask.any():
            return None

        # Extract indices where the condition is true
        result_indices = torch.nonzero(mask)[:, 1]
        return result_indices

    def get_bounded_scaler(self):
        if self.p_model is None:
            return torch.ones(self.p_target.shape[0]).to(self.p_target.device)
        else:
            # Calculate the scaling factor and apply bound;
            n_classes = self.p_model.shape[0]
            kl = torch.log(1. * torch.tensor([n_classes]).to(self.p_target.device)) + torch.special.entr(self.p_model).sum()

            lower_bound = 1. / (1. + kl)
            upper_bound = 1. + kl

            s = (self.p_target + 1e-6) / (self.p_model + 1e-6)

            s_bounded = torch.clamp(s, min=lower_bound, max=upper_bound)

            return s_bounded


class OBSHook(Hook):
    def __init__(self, obs_target_dict):

        self.obs_target = obs_target_dict

        self.obs_target_cnt = {}
        for k, v in self.obs_target.items():
            self.obs_target_cnt[k] = 0

    # TODO: the actual training times differences between the labeled instances and the ulb instances might be a problem;


    @torch.no_grad()
    def before_train_step(self, algorithm, *args, **kwargs):
        # Process and save every obs_freq (default: 256) iterations;
        if (self.every_n_iters(algorithm, algorithm.num_log_iter) or algorithm.it == 0) \
                and algorithm.it < algorithm.num_train_iter:
            i = algorithm.it // algorithm.num_log_iter
            print('Saving obs at the {}th checkpoint'.format(i))

            # Update and save p_model;
            p_model = algorithm.hooks_dict['UAWeightingHook'].p_model
            if p_model is not None:
                self.update_and_save_obs_target(algorithm, {'p_model': p_model.detach().cpu().numpy()})

    def update_and_save_obs_target(self, algorithm, target_dict):
        for target_name, target in target_dict.items():
            i = self.obs_target_cnt[target_name]
            self.obs_target[target_name][i] = target
            self.obs_target_cnt[target_name] += 1

            save_fp = algorithm.save_dir + '/{}/{}.pkl'.format(algorithm.save_name, target_name)
            with open(save_fp, 'wb') as f:
                pickle.dump(self.obs_target[target_name], f)



