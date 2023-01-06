import pickle
import time
import mmcv
import torch
import warnings
import copy

import os.path as osp 
import torch.distributed as dist
import numpy as np

from mmcv.runner import HOOKS, Hook, EvalHook, get_dist_info, NeptuneLoggerHook
from neptune.new.types import File

from .utils import  (get_mistakes_summary, get_text_summary_mistakes, show_mistakes_ids_pct, get_metrics_summary, \
    get_metrics_from_summary, plot_track_length_frequency, show_metrics_decisions, show_metrics_dec_pct)
            


@HOOKS.register_module()
class CustomEval(Hook):

    def __init__(self,interval,eval_at_zero=False,eval_start_epoch=0):
        self.interval = interval
        self.tmpdir = None
        self.eval_at_zero = eval_at_zero
        self.eval_start_epoch = eval_start_epoch

    def after_run(self, runner):
        self.validation_step(runner)

    def before_epoch(self, runner):
        if runner._epoch < self.eval_start_epoch:
            return
        elif ( runner._epoch % self.interval ) == 0 and runner._epoch > 0:
            self.validation_step(runner)
        elif runner._epoch == 0 and self.eval_at_zero:
            self.validation_step(runner)

    def log_tracking_eval_to_neptune(self,runner,neptune,metrics_summary,prefix='overall/'):
        """Method lo log different outputs to Neptune."""
        for k,v in metrics_summary.items():
            if type(v) == dict:
                # print("pass",k)
                self.log_tracking_eval_to_neptune(runner,neptune,v,prefix=prefix+k+'/')
            else:
                if str(v) == 'nan':
                    v = -1
                neptune[prefix + k].log(v, step=runner._epoch)

    def log_eval_to_neptune(self,runner,neptune,all_log_vars):
        """Method lo log different outputs to Neptune."""
        mistakes_summary = get_mistakes_summary(all_log_vars)
        plots = []
        get_text_summary_mistakes(mistakes_summary)
        plots += show_mistakes_ids_pct(mistakes_summary)
        tracker = runner.model.module.trackManager.tracker

        metrics_summary = get_metrics_summary(all_log_vars)
        metrics = get_metrics_from_summary(metrics_summary,
                                           suffix=tracker.cls,
                                           detection_decisions=tracker.detection_decisions,
                                           tracking_decisions=tracker.tracking_decisions)

        plots += plot_track_length_frequency(metrics_summary)
        plots += show_metrics_decisions(metrics)
        plots += show_metrics_dec_pct(metrics)


        
        for f in plots:
            neptune['mistakes_and_metrics'].log(File(f))


    def validation_step(self, runner):
        for hook in runner._hooks:
            if issubclass(type(hook), EvalHook):
                dataloader = hook.dataloader
            elif issubclass(type(hook), NeptuneLoggerHook):
                neptune = hook.run


        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.custom_hook')

        rank, world_size = get_dist_info()
        if world_size == 1:
            results, all_log_vars = single_gpu_test_simple(runner.model, dataloader)
            print('\n\nLogging all vars to all_log_vars.pkl\n ')
            pickle.dump(all_log_vars, open('all_log_vars.pkl','wb'))
            self.log_eval_to_neptune(runner,neptune,all_log_vars)


            train = False if 'val' in dataloader.dataset.dataset.ann_file else True
            if results == None: 
                pass
            else:

                # self.log_eval_to_neptune(runner,neptune,all_log_vars,prefix='validation_')
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    metrics_summary, nusc_results_trk = dataloader.dataset.evaluate_tracking(copy.deepcopy(results),train=train,neptune=neptune)
                del metrics_summary['meta']
                del metrics_summary['cfg']
                self.log_tracking_eval_to_neptune(runner,neptune,metrics_summary,prefix='trk-metric: overall/')

                metrics_summary, nusc_results_det = dataloader.dataset.evaluate_detection(results,train=train)
                del metrics_summary['meta']
                del metrics_summary['cfg']
                self.log_tracking_eval_to_neptune(runner,neptune,metrics_summary,prefix='det-metric: overall/')
        else:
            results, all_log_vars = multi_gpu_test(runner.model, dataloader, tmpdir=tmpdir, gpu_collect=True)
            print("\n[after multi_gpu_test] current runner.rank:{}".format(runner.rank))
            
            if runner.rank == 0:

                print('\n\nLogging all vars to all_log_vars.pkl\n ')
                pickle.dump(all_log_vars, open('all_log_vars.pkl','wb'))
                self.log_eval_to_neptune(runner,neptune,all_log_vars)


                train = False if 'val' in dataloader.dataset.dataset.ann_file else True
                if results == None: 
                    pass
                else:

                    # self.log_eval_to_neptune(runner,neptune,all_log_vars,prefix='validation_')
                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        metrics_summary, nusc_results_trk = dataloader.dataset.evaluate_tracking(copy.deepcopy(results),train=train,neptune=neptune)
                    del metrics_summary['meta']
                    del metrics_summary['cfg']
                    self.log_tracking_eval_to_neptune(runner,neptune,metrics_summary,prefix='trk-metric: overall/')

                    metrics_summary, nusc_results_det = dataloader.dataset.evaluate_detection(results,train=train)
                    del metrics_summary['meta']
                    del metrics_summary['cfg']
                    self.log_tracking_eval_to_neptune(runner,neptune,metrics_summary,prefix='det-metric: overall/')
            
            print('before dist barrier')
            dist.barrier()

        print('after dist barrier, rank:',rank)
     



def single_gpu_test_simple(model,data_loader):
    """Test model with single gpu.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list[dict]: The prediction results.
    """
    results = []
    all_log_vars = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, log_vars = model(return_loss=False, rescale=True, **data)
            
        temp_idx = []
        for ii,x in enumerate(result):
            if x['pts_bbox'] == False:#drop padded entries of sequences
                pass
            else:
                temp_idx.append(ii)

        if len(temp_idx) != len(result): #skip padded predictions
            result = [result[idx] for idx in temp_idx]
            log_vars = [log_vars[idx] for idx in temp_idx]
            
            
            
        results.extend(result)
        all_log_vars.extend(log_vars)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results, all_log_vars


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    all_log_vars = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, log_vars = model(return_loss=False, rescale=True, **data)
            # print(log_vars)

        batch_size = len(result)
        temp_idx = []
        for ii,x in enumerate(result):
            if x['pts_bbox'] == False: #drop padded entries of sequences
                pass
            else:
                temp_idx.append(ii)
                

        if len(temp_idx) != len(result): #skip padded predictions
            result = [result[idx] for idx in temp_idx]
            log_vars = [log_vars[idx] for idx in temp_idx]
        
        results.extend(result)
        all_log_vars.extend(log_vars)

        # print(type(results))

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()


        # if i > 100: 
        #     break


    dist.barrier()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
        dist.barrier()
        all_log_vars = collect_results_gpu(all_log_vars, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)


    return results, all_log_vars



def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()

    # print(result_part)
    # dump result part to tensor with pickle

    b = bytearray(pickle.dumps(result_part))
    part_tensor = torch.tensor(b, dtype=torch.uint8, device='cuda')
    # part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        # for res in zip(*part_list):
        #     ordered_results.extend(list(res))
        for res in part_list:
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        # ordered_results = ordered_results[:size]

        return ordered_results




import tempfile
import shutil

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results