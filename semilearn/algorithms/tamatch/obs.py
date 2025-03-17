import torch
import torch.nn as nn

import pickle

import numpy as np
from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import concat_all_gather
from torch.nn import functional as F



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