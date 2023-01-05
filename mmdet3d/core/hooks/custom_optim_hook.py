from mmcv.runner import HOOKS, Hook, get_dist_info
from mmcv.runner.hooks.optimizer import OptimizerHook
import torch.distributed as dist
import time
import torch

@HOOKS.register_module()
class CustomOptimHook(OptimizerHook):
    """Hook to prevent bug when training tracker with momentum otpimizers"""

    def __init__(self,accumulate=False,accumulate_num=0,verbose=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.accumulate = accumulate
        self.accumulate_num = accumulate_num
        self.step = False
        self.verbose = verbose
        self.rank, self.world_size = get_dist_info()
        if self.verbose:
            print("[CustomOptimHook] accumulate: {}, accumulate_num: {}, verbose: {}".format(accumulate,accumulate_num,verbose))

        self.count = 0

    def log_msg(self,runner):
        return "[CustomOptimHook| rank:{},epoch:{},iter:{}]".format(runner.rank,runner._epoch,runner._iter)

    def after_train_iter(self, runner):
        if self.verbose:
            t1 = time.time()
            print("{} Starting after_train_iter()".format(self.log_msg(runner)))

        if runner.model.module.take_step_now():
            if self.accumulate:
                self.after_train_iter_accum(runner)
            else:
                self.after_train_iter_(runner)
                
        if self.verbose:
            print("{} Ending after_train_iter() after {}s".format(self.log_msg(runner),time.time()-t1))


    def after_train_iter_(self, runner):
        if self.verbose:
            t1 = time.time()
            print("{} Starting after_train_iter_()".format(self.log_msg(runner)))

        # for name,p in runner.model.module.named_parameters():
        #     if p.grad == None and p.requires_grad == True:
        #         p.grad = 0.0 * p.data


        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        if self.world_size > 1:
            if self.verbose:
                print("{} - dist.barrier()".format(self.log_msg(runner)))
            dist.barrier()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
                            
        runner.optimizer.step()

        if self.verbose:
            print("{} Ending after_train_iter_() after {}s".format(self.log_msg(runner),time.time()-t1))








    def after_train_iter_accum(self, runner):
        if self.verbose:
            t1 = time.time()
            print("{} Starting after_train_iter_accum()".format(self.log_msg(runner)))

        if self.step == 0:
            runner.optimizer.zero_grad()

        runner.outputs['loss'].backward()
        
        if self.world_size > 1:
            if self.verbose:
                print("{} - dist.barrier()".format(self.log_msg(runner)))
            dist.barrier()
        

        if self.step == self.accumulate_num - 1:


            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])

            
            runner.optimizer.step()

            self.step = 0
        else:
            self.step += 1

        if self.verbose:
            print("{} Ending after_train_iter_accum() after {}s".format(self.log_msg(runner),time.time()-t1))
