from mmcv.runner import HOOKS, Hook, NeptuneLoggerHook


@HOOKS.register_module()
class UpdateGradMonitor(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        self.update_grad_monitor(runner)

    def update_grad_monitor(self,runner):
        neptune = None
        for hook in runner._hooks:
            if issubclass(type(hook), NeptuneLoggerHook):
                neptune = hook.run
                
        runner.model.module.trackManager.update_grad_monitor(neptune)


@HOOKS.register_module()
class UpdateParamMonitor(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        self.update_param_monitor(runner)

    def update_param_monitor(self,runner):
        neptune = None
        for hook in runner._hooks:
            if issubclass(type(hook), NeptuneLoggerHook):
                neptune = hook.run

        runner.model.module.trackManager.update_param_monitor(neptune)
        





