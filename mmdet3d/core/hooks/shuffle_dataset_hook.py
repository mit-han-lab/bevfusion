from mmcv.runner import HOOKS, Hook
import torch
import random
import numpy as np


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@HOOKS.register_module()
class ShuffleDatasetHook(Hook):
    """Hook to change the shuffle seed at each epoch"""

    def after_epoch(self, runner):
        seed = runner.meta['seed'] + runner._epoch
        set_random_seed(seed)
        runner.data_loader.dataset.shuffle_sequences()
        
