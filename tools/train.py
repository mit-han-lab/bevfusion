import argparse
import copy
import os
import random
import time

import os.path as osp
import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval



def setup_neptune_logger(cfg,args):
    try:
        cfg.dataloader_kwargs
    except AttributeError:
        raise AttributeError("You need to add 'dataloader_kwargs' to your config file. See 'configs/_base_/reidentification_runtine.py' for an example.")

    try:
        cfg.train_tracker
    except AttributeError:
        raise AttributeError("You need to add 'train_tracker' to your config file. See 'configs/_base_/reidentification_runtine.py' for an example.")
        

    if cfg.train_tracker:
        assert cfg.dataloader_kwargs['shuffle'] == False, "You need to set 'dataloader_kwargs.shuffle' to False when training a tracker."

    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # neptune logging setup
    try:
        if cfg.log_config.hooks[1].type == 'NeptuneLoggerHook':
            source_files = [x.strip() for x in cfg._text.split("\n") if x.strip().endswith(".py")]
            cfg.log_config.hooks[1].init_kwargs['source_files'] = source_files
            
            schedule_file = [x for x in source_files if "schedule" in x][0]
            dataset_file = [x for x in source_files if "/datasets/" in x][0]

            if 'medium' in dataset_file:
                dataset = 'v1.0-medium'
            elif 'mini' in dataset_file:
                dataset = 'v1.0-mini'
            else:
                dataset = 'v1.0-trainval'


            if cfg.log_config.hooks[1].init_kwargs.project == "bentherien/re-identification":
                neptune_name = "[Schedule] {} [Config] {}".format(osp.basename(schedule_file), 
                                                                                        osp.basename(args.config),)
            else:
                model_file = [x for x in source_files if "pnp_net" in x][-1]
                neptune_name = "[Schedule] {} [Model] {} [Config] {} [gpus] {}".format(osp.basename(schedule_file), 
                                                                                        osp.basename(model_file), 
                                                                                        osp.basename(args.config),
                                                                                        cfg.data.train.gpus)
                                                                                        

            cfg.log_config.hooks[1].init_kwargs['name'] += neptune_name
            cfg.log_config.hooks[1].init_kwargs['tags'] += [dataset] + cfg.neptune_tags 
            

            print(cfg.log_config.hooks[1].init_kwargs['name'])

            
        else:
            print('###############################################################################################')
            print('\t WARNING : No NeptuneLoggerHook in config file. This run will not be logged to Neptune.')
            print('###############################################################################################')
            time.sleep(3)

    except IndexError:

        pass

    return cfg


def main():

    assert '/btherien/github/nuscenes-devkit/python-sdk' in os.environ['PYTHONPATH']
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()


    if 'tracking' in args.config.split('/')[0]:
        cfg = Config.fromfile(args.config)
        cfg = setup_neptune_logger(cfg,args)
        dataloader_kwargs=cfg.dataloader_kwargs
    else:
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg = Config(recursive_eval(configs), filename=args.config)
        dataloader_kwargs=dict(shuffle=True, prefetch_factor=4)
    
    # print(cfg.pretty_text)
    # exit(0)
    
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    print("\n\n")
    print("###############################################################################################")
    print("Setting local rank to {}".format(dist.local_rank()))
    print("###############################################################################################")
    print("\n\n")
    time.sleep(0.2)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # print(cfg.data.train)
    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
        dataloader_kwargs=dataloader_kwargs,
    )


if __name__ == "__main__":
    main()
