import torch
import numpy as np
from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import concat_all_gather


class TAHook(Hook):
    """
    TAMatch Hook: rescaling and reweighting for the model's raw prediction on unlabeled data;
    1. Initialization: set the p_target:
        a. uniform: a uniform categorical distribution;
        b. prior: a given prior categorical distribution;
        c. model: start from uniform and update the model's prediction on ulb data with ema;
            Note: ulb actually contains the lb data in most settings be default in USB;
    2. Update: in each iteration, given raw prediction: p_x_ulb (p_model):
        a. calculate the per-class scaling factor: r = p_target / p_model;
        b. optional: calculate the bound and clip the scaling factor;
        c. rescaling: p_x_ulb_new = Normalize(p_ulb / p_model);
        d. calculate the pseudo labels (pl) with p_x_ulb_new (outside this hook);
        e. reweighting: calculate the weighting mask for each pl;

    Also provide control to turn off rescale or reweight:
        a. rescale on, reweight on: TAMatch;
        b. rescale off, reweight off: FixMatch;
        c. rescale on, reweight off: FixMatch + dist_alignment (slight difference);
        d. rescale off, reweight on: V4;
    """

    def __init__(
            self,
            num_classes,
            rescale=True,
            reweight=True,
            p_model_momentum=0.999,
            p_target_momentum=0.99999,
            p_target_type='uniform',
            p_target=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.rescale = rescale
        self.reweight = reweight
        self.p_target_m = p_target_momentum
        self.p_model_m = p_model_momentum

        # Set the p_target;
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)
        # Initialize the p_model as None (the first prediction);
        self.p_model = None

    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'prior', 'model']

        # Unless set with 'model', p_target will not be updated;
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes,)) / self.num_classes
            update_p_target = True
        else:
            # Set p_target with a prior distribution;
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)

        return update_p_target, p_target

    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb):
        # Check device;
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        # Update p_model with batch p_x_ulb;
        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model is None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.p_model_m + torch.mean(probs_x_ulb, dim=0) * (1. - self.p_model_m)

        # Update p_target with p_model (ema estimated version) with a slow ema;
        if self.update_p_target:
            self.p_target = self.p_target * self.p_target_m + torch.mean(self.p_model, dim=0) * (1. - self.p_target_m)

    @torch.no_grad()
    def debias(self, algorithm, probs_x_ulb):
        #  Update p_model (and p_target);
        self.update_p(algorithm, probs_x_ulb)

        # Calculate the scaling factor;
        r = (self.p_target + 1e-6) / (self.p_model + 1e-6)

        # Rescaling;
        if self.rescale:
            probs_x_ulb_debiased = probs_x_ulb * r
            probs_x_ulb_debiased = probs_x_ulb_debiased / probs_x_ulb_debiased.sum(dim=-1, keepdim=True)
        else:
            probs_x_ulb_debiased = probs_x_ulb

        # Reweighting;
        if self.reweight:
            # Get the predicted class of unlabeled data;
            # Note that unqualified pl will be filtered by the binary maks generated from the pseudo-labeling hook;
            plb = torch.argmax(probs_x_ulb, dim=-1)
            weight_mask = r[plb]
        else:
            weight_mask = torch.ones(self.num_classes).to(probs_x_ulb.device)

        return probs_x_ulb_debiased, weight_mask









