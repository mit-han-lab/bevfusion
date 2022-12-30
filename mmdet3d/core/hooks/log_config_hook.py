from mmcv.runner import HOOKS, Hook, NeptuneLoggerHook


@HOOKS.register_module()
class UploadConfig(Hook):
    """Hook to log the configuration to netptune"""

    def __init__(self,cfg_path):
        self.cfg_path = cfg_path
        self.uploaded = False


    def upload_cfg(self, runner):
        if runner.rank == 0:
            for hook in runner._hooks:
                if issubclass(type(hook), NeptuneLoggerHook):
                    neptune = hook.run

            neptune['config_file'].upload(self.cfg_path)

    def before_run(self, runner):
        if not self.uploaded:
            self.upload_cfg(runner)
            self.uploaded = True
        

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

