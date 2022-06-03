import copy
import torch
from collections import deque


__all__ = ["convert_sync_batchnorm"]


def convert_sync_batchnorm(input_model, exclude=[]):
    for name, module in input_model._modules.items():
        skip = sum([ex in name for ex in exclude])
        if skip:
            continue
        input_model._modules[name] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
    return input_model
    