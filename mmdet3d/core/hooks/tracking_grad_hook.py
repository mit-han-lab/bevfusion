from mmcv.runner import HOOKS, Hook

#unused
@HOOKS.register_module()
class TrackingGradHook(Hook):
    """Hook to prevent bug when training tracker with momentum otpimizers"""

    def __init__(self):
        pass

    def after_iter(self, runner):
        for param in runner.model.parameters():
            param.grad = None

