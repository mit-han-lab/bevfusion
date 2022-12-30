# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHookTracking(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch=runner.epoch, max_epoch=runner.max_epochs)