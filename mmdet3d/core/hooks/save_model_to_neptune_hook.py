import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, EvalHook, get_dist_info, NeptuneLoggerHook


@HOOKS.register_module()
class SaveModelToNeptuneHook(Hook):

    def __init__(self):
        pass

    def after_run(self, runner):
        self.save_model_neptune(runner)

    def save_model_neptune(self,runner):
        rank, world_size = get_dist_info()
        if world_size == 1:
                self.save_model_neptune_(runner)
        else:
            print("[Info] in save_model_neptune()")
            if runner.rank == 0:
                self.save_model_neptune_(runner)
            dist.barrier()

    def save_model_neptune_(self,runner):
        print('Saving model in SaveModelToNeptuneHook')
        neptune = None
        for hook in runner._hooks:
            if issubclass(type(hook), NeptuneLoggerHook):
                neptune = hook.run

        runner.save_checkpoint(out_dir="/tmp",
                                filename_tmpl = 'epoch_{}.pth',
                                save_optimizer = True,
                                meta = None,
                                create_symlink = False) # save checkpoint to /tmp
        print("[Info] Uploading model to Neptune...")
        neptune["model/saved_model"].upload("/tmp/epoch_{}.pth".format(runner.epoch + 1))
        print("[Info] Save Complete")

        

# not used
@HOOKS.register_module()
class LogConfigToNeptuneHook(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        self.save_model_neptune(runner)
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
        pass

    def log_config(self,runner):
        rank, world_size = get_dist_info()
        if world_size == 1:
            self.log_config_()
           
        else:
            if runner.rank == 0:
                self.log_config_(runner)
            dist.barrier()

    def log_config_(self,runner):
        neptune = None
        for hook in runner._hooks:
            if issubclass(type(hook), NeptuneLoggerHook):
                neptune = hook.run

        neptune['iters'].log(runner._max_iters)
        neptune['epochs'].log(runner._max_epochs)




