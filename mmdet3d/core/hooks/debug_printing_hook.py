from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DebugPrintingHook(Hook):
    """Hook to print messages for debugging"""

    def __init__(self):
        pass
    
    def log_msg(self,runner):
        return "[DebugPrintingHook| rank:{},epoch:{},iter:{},Priority:{}]".format(runner.rank,runner._epoch,runner._iter,self.priority)

    def before_run(self, runner):
        print("{} before_run()".format(self.log_msg(runner)))

    def after_run(self, runner):
        print("{} after_run()".format(self.log_msg(runner)))

    def before_epoch(self, runner):
        print("{} before_epoch()".format(self.log_msg(runner)))

    def after_epoch(self, runner):
        print("{} after_epoch()".format(self.log_msg(runner)))

    def before_iter(self, runner):
        print("{} before_iter()".format(self.log_msg(runner)))

    def after_iter(self, runner):
        print("{} after_iter()".format(self.log_msg(runner)))
