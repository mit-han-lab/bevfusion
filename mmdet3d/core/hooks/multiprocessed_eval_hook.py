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

def add_mistakes_track(metrics_summary,k,v):
    for k1,v1 in v:
        try:
            metrics_summary[k][k1]['count'] += v1['count']
            metrics_summary[k][k1]['cost_diff'] += v1['cost_diff']
            metrics_summary[k][k1]['tp_diff_max'] += v1['tp_diff_max']
            try:
                metrics_summary[k][k1]['track_type']['FP'] += v1['track_type']['FP']
                metrics_summary[k][k1]['track_type']['TP'] += v1['track_type']['TP']
                metrics_summary[k][k1]['track_type']['TP-OOR'] += v1['track_type']['TP-OOR']
                
            except KeyError:
                metrics_summary[k][k1]['track_type'] = v1['track_type']
        except KeyError:
            metrics_summary[k][k1] = v1



def add_mistakes_det(metrics_summary,k,v):
    for k1,v1 in v:
        try:
            metrics_summary[k][k1]['count'] += v1['count']
            metrics_summary[k][k1]['cost_diff'] += v1['cost_diff']
            metrics_summary[k][k1]['tp_diff_max'] += v1['tp_diff_max']
            try:
                metrics_summary[k][k1]['det_type']['FP'] += v1['det_type']['FP']
                metrics_summary[k][k1]['det_type']['TP'] += v1['det_type']['TP']
                
            except KeyError:
                metrics_summary[k][k1]['det_type'] = v1['det_type']
        except KeyError:
            metrics_summary[k][k1] = v1
            


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

    def log_eval_to_neptune(self,runner,neptune,all_log_vars,prefix='validation'):
        """Method lo log different outputs to Neptune."""
        # print(all_log_vars)
        metrics = [{k[5:]:v for k,v in x.items() if k.startswith('eval_')} for x in all_log_vars]
        metrics_summary = {}
        for x in metrics:
            for k,v in x.items():
                if k.startswith('mistakes_track_'):
                    add_mistakes_track(metrics_summary,k,v)
                elif k.startswith('mistakes_det_'):
                    add_mistakes_det(metrics_summary,k,v)

                elif type(v) == list:
                    try:
                        metrics_summary[k].extend(v)
                    except KeyError:
                        metrics_summary[k] = v

                elif type(v) == int or type(v) == float or type(v) == np.float64:
                    if k.startswith('scene_'):
                        try:
                            metrics_summary[k[6:]].append(v)
                        except KeyError:
                            metrics_summary[k[6:]] = [v]
                    else:
                        try:
                            metrics_summary[k] += v
                        except KeyError:
                            metrics_summary[k] = v
                else:
                    print(type(v))


        for k,v in metrics_summary.items():
            if type(v) == list and type(v[0]) == torch.Tensor:
                metrics_summary[k] = torch.cat(v)

        tracker = runner.model.module.trackManager.trackers['car']

        metrics = {}
        metrics[f'mean_track_length_{tracker.cls}'] = np.mean(metrics_summary[f'len_tracks'])
        metrics[f'median_track_length_{tracker.cls}'] = np.median(metrics_summary[f'len_tracks'])
        metrics[f'mean_track_length_>1_{tracker.cls}'] = np.mean(metrics_summary[f'greater_than_one_tracks'])
        metrics[f'Mean_tracks_per_scene_{tracker.cls}'] = np.mean(metrics_summary[f'total_tracks'])

        for k in ['total','det_match'] + tracker.detection_decisions + tracker.tracking_decisions:
            if k == 'total':
                metrics[f'acc_{k}_{tracker.cls}'] = metrics_summary[k+'_correct'] / (metrics_summary[k+'_gt'] + 0.000000000001)
            else:
                metrics[f'recall_{k}_{tracker.cls}'] = metrics_summary[k+'_correct'] / ( metrics_summary[k+'_gt'] + 0.000000000001)
                metrics[f'precision_{k}_{tracker.cls}'] = metrics_summary[k+'_correct'] / ( metrics_summary[k+'_num_pred'] + 0.000000000001)

                add = metrics[f'recall_{k}_{tracker.cls}'] + metrics[f'precision_{k}_{tracker.cls}'] + 0.000000000001
                mul = metrics[f'recall_{k}_{tracker.cls}'] * metrics[f'precision_{k}_{tracker.cls}']
                metrics[f'f1_{k}_{tracker.cls}'] = 2 * (mul/add)
                

        total_tp_decisions = torch.cat([x for k,x in metrics_summary.items() if k.startswith('num_TP')]).sum()
        for k in metrics_summary:

            if k.startswith('num_TP') or k.startswith('num_dec'):
                metrics[f'%_{k[4:]}'] = metrics_summary[k].sum() / (total_tp_decisions + 0.000000000001)

            elif not ( k.endswith('_gt') or k.endswith('_correct') or k.endswith('_num_pred')):
                try:
                    metrics[k] = metrics_summary[k].mean().item()
                except AttributeError:
                    metrics[k] = np.mean(metrics_summary[k])

        

        total_ids = torch.sum([x for k,x in metrics_summary.items() if k.startswith('IDS_')])
        metrics["total_IDS"] = total_ids
        for k,x in metrics_summary.items():
            if k.startswith('IDS_'):
                metrics[k] = torch.sum(x) / (total_ids + 0.000000000001)

        
        total_mistakes_track = torch.sum([y['count'] for k,x in metrics_summary.items() if k.startswith('mistakes_track_') for y in x.values()])
        metrics["total_mistakes_track"] = total_mistakes_track


        mistakes_track_summary = {}
        mistakes_det_summary = {}

        for k,x in metrics_summary.items():
            if k.startswith('mistakes_track_'):
                for k1,v1 in x.items():
                    try:
                        mistakes_track_summary[k]['count'] += v1['count']
                        mistakes_track_summary[k]['cost_diff'] += v1['cost_diff']
                        mistakes_track_summary[k]['tp_diff_max'] += v1['tp_diff_max']
                        try:
                            mistakes_track_summary[k]['track_type']['FP'] += v1['track_type']['FP']
                            mistakes_track_summary[k]['track_type']['TP'] += v1['track_type']['TP']
                            mistakes_track_summary[k]['track_type']['TP-OOR'] += v1['track_type']['TP-OOR']
                            
                        except KeyError:
                            mistakes_track_summary[k]['track_type'] = v1['track_type']
                    except KeyError:
                        mistakes_track_summary[k] = v1
            elif k.startswith('mistakes_det_'):
                for k1,v1 in x.items():
                    try:
                        mistakes_det_summary[k]['count'] += v1['count']
                        mistakes_det_summary[k]['cost_diff'] += v1['cost_diff']
                        mistakes_det_summary[k]['tp_diff_max'] += v1['tp_diff_max']
                        try:
                            mistakes_det_summary[k]['det_type']['FP'] += v1['det_type']['FP']
                            mistakes_det_summary[k]['det_type']['TP'] += v1['det_type']['TP']
                            
                        except KeyError:
                            mistakes_det_summary[k]['det_type'] = v1['det_type']
                    except KeyError:
                        mistakes_det_summary[k] = v1



        for k,x in metrics_summary.items():
            if k.startswith('mistakes_track_'):
                for k1,v1 in x.items():
                    pass


        for k,x in metrics_summary.items():
            if k.startswith('mistakes_det_'):
                for k1,v1 in x.items():
                    pass
                
        mistakes_det_summary = {}



        for k,v in metrics.items():
            if str(v) == 'nan':
                continue
            
            neptune[f'{prefix}{k}'].log(v, step=runner._epoch)


        

        
        # exit(0)

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
            # print("\n[after multi_gpu_test] current runner.rank:{} resulta:{} ".format(runner.rank,results))
            
            if runner.rank == 0:
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
            
            dist.barrier()
     



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


    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
        all_log_vars = collect_results_gpu(all_log_vars, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)


    return results, all_log_vars



def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
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