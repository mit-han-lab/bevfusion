# BEVFusion

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevfusion-multi-task-multi-sensor-fusion-with/3d-object-detection-on-nuscenes)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes?p=bevfusion-multi-task-multi-sensor-fusion-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevfusion-multi-task-multi-sensor-fusion-with/3d-multi-object-tracking-on-nuscenes)](https://paperswithcode.com/sota/3d-multi-object-tracking-on-nuscenes?p=bevfusion-multi-task-multi-sensor-fusion-with)

### [website](http://bevfusion.mit.edu/) | [paper](https://arxiv.org/abs/2205.13542) | [video](https://www.youtube.com/watch?v=uCAka90si9E)

![demo](assets/demo.gif)

## News

**If you are interested in getting updates, please sign up [here](https://docs.google.com/forms/d/e/1FAIpQLSfkmfsX45HstL5rUQlS7xJthhS3Z_Pm2NOVstlXUqgaK4DEfQ/viewform) to get notified!**

- **(2023/1/16)** BEVFusion is accepted to ICRA 2023!
- **(2022/8/16)** BEVFusion ranks first on [Waymo](https://waymo.com/open/challenges/2020/3d-detection/) 3D object detection leaderboard among all solutions.
- **(2022/6/3)** BEVFusion ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions.
- **(2022/6/3)** We released the first version of BEVFusion (with pre-trained checkpoints and evaluation).
- **(2022/5/26)** BEVFusion is released on [arXiv](https://arxiv.org/abs/2205.13542).
- **(2022/5/2)** BEVFusion ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions that do not use test-time augmentation and model ensemble.

## Abstract

Multi-sensor fusion is essential for an accurate and reliable autonomous driving system. Recent approaches are based on point-level fusion: augmenting the LiDAR point cloud with camera features. However, the camera-to-LiDAR projection throws away the semantic density of camera features, hindering the effectiveness of such methods, especially for semantic-oriented tasks (such as 3D scene segmentation). In this paper, we break this deeply-rooted convention with BEVFusion, an efficient and generic multi-task multi-sensor fusion framework. It unifies multi-modal features in the shared bird's-eye view (BEV) representation space, which nicely preserves both geometric and semantic information. To achieve this, we diagnose and lift key efficiency bottlenecks in the view transformation with optimized BEV pooling, reducing latency by more than **40x**. BEVFusion is fundamentally task-agnostic and seamlessly supports different 3D perception tasks with almost no architectural changes. It establishes the new state of the art on the nuScenes benchmark, achieving **1.3%** higher mAP and NDS on 3D object detection and **13.6%** higher mIoU on BEV map segmentation, with **1.9x** lower computation cost.

## Results

### 3D Object Detection (on Waymo test)

|   Model   | mAP-L1 | mAPH-L1  | mAP-L2  | mAPH-L2  |
| :-------: | :------: | :--: | :--: | :--: |
| [BEVFusion](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&challengeId=DETECTION_3D&emailId=f58eed96-8bb3&timestamp=1658347965704580) |    82.72   |  81.35  | 77.65  |  76.33 |
| [BEVFusion-TTA](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&challengeId=DETECTION_3D&emailId=94ddc185-d2ce&timestamp=1663562767759105) | 86.04    |  84.76 | 81.22  |  79.97 |

Here, BEVFusion only uses a single model without any test time augmentation. BEVFusion-TTA uses single model with test-time augmentation and no model ensembling is applied. 

### 3D Object Detection (on nuScenes test)

|   Model   | Modality | mAP  | NDS  |
| :-------: | :------: | :--: | :--: |
| BEVFusion-e |   C+L    | 74.99 | 76.09 |
| BEVFusion |   C+L    | 70.23 | 72.88 |
| BEVFusion-base* |   C+L    | 71.72 | 73.83 |

*: We scaled up MACs of the model to match the computation cost of concurrent work.

### 3D Object Detection (on nuScenes validation)

|        Model         | Modality | mAP  | NDS  | Checkpoint  |
| :------------------: | :------: | :--: | :--: | :---------: |
|    [BEVFusion](configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml)       |   C+L    | 68.52 | 71.38 | [Link](https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-det.pth) |
| [Camera-Only Baseline](configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml) |    C     | 35.56 | 41.21 | [Link](https://bevfusion.mit.edu/files/pretrained_updated/camera-only-det.pth) |
| [LiDAR-Only Baseline](configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml)  |    L     | 64.68 | 69.28 | [Link](https://bevfusion.mit.edu/files/pretrained/lidar-only-det.pth) |

*Note*: The camera-only object detection baseline is a variant of BEVDet-Tiny with a much heavier view transformer and other differences in hyperparameters. Thanks to our [efficient BEV pooling](mmdet3d/ops/bev_pool) operator, this model runs fast and has higher mAP than BEVDet-Tiny under the same input resolution. Please refer to [BEVDet repo](https://github.com/HuangJunjie2017/BEVDet) for the original BEVDet-Tiny implementation. The LiDAR-only baseline is TransFusion-L.

### BEV Map Segmentation (on nuScenes validation)

|        Model         | Modality | mIoU | Checkpoint  |
| :------------------: | :------: | :--: | :---------: |
| [BEVFusion](configs/nuscenes/seg/fusion-bev256d2-lss.yaml)       |   C+L    | 62.95 | [Link](https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-seg.pth) |
| [Camera-Only Baseline](configs/nuscenes/seg/camera-bev256d2.yaml) |    C     | 57.09 | [Link](https://bevfusion.mit.edu/files/pretrained_updated/camera-only-seg.pth) |
| [LiDAR-Only Baseline](configs/nuscenes/seg/lidar-centerpoint-bev128.yaml)  |    L     | 48.56 | [Link](https://bevfusion.mit.edu/files/pretrained/lidar-only-seg.pth) |

## Usage

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

We also provide a [Dockerfile](docker/Dockerfile) to ease environment setup. To get started with docker, please make sure that `nvidia-docker` is installed on your machine. After that, please execute the following command to build the docker image:

```bash
cd docker && docker build . -t bevfusion
```

We can then run the docker with the following command:

```bash
nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g bevfusion /bin/bash
```

We recommend the users to run data preparation (instructions are available in the next section) outside the docker if possible. Note that the dataset directory should be an absolute path. Within the docker, please run the following command to clone our repo and install custom CUDA extensions:

```bash
cd home && git clone https://github.com/mit-han-lab/bevfusion && cd bevfusion
python setup.py develop
```

You can then create a symbolic link `data` to the `/dataset` directory in the docker.

### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### Evaluation

We also provide instructions for evaluating our pretrained models. Please download the checkpoints using the following script: 

```bash
./tools/download_pretrained.sh
```

Then, you will be able to run:

```bash
torchpack dist-run -np 8 python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval [evaluation type]
```

For example, if you want to evaluate the detection variant of BEVFusion, you can try:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
```

While for the segmentation variant of BEVFusion, this command will be helpful:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
```

### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train the camera-only variant for object detection, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For camera-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/camera-bev256d2.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For LiDAR-only detector, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

For LiDAR-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/lidar-centerpoint-bev128.yaml
```

For BEVFusion detection model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth 
```

For BEVFusion segmentation model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

Note: please run `tools/test.py` separately after training to get the final evaluation metrics.

## FAQs

Q: Can we directly use the info files prepared by mmdetection3d?

A: We recommend re-generating the info files using this codebase since we forked mmdetection3d before their [coordinate system refactoring](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/changelog.md).

## Acknowledgements

BEVFusion is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [BEVDet](https://github.com/HuangJunjie2017/BEVDet), [TransFusion](https://github.com/XuyangBai/TransFusion), [CenterPoint](https://github.com/tianweiy/CenterPoint), [MVP](https://github.com/tianweiy/MVP), [FUTR3D](https://arxiv.org/abs/2203.10642), [CVT](https://github.com/bradyz/cross_view_transformers) and [DETR3D](https://github.com/WangYueFt/detr3d). 

Please also check out related papers in the camera-only 3D perception community such as [BEVDet4D](https://arxiv.org/abs/2203.17054), [BEVerse](https://arxiv.org/abs/2205.09743), [BEVFormer](https://arxiv.org/abs/2203.17270), [M2BEV](https://arxiv.org/abs/2204.05088), [PETR](https://arxiv.org/abs/2203.05625) and [PETRv2](https://arxiv.org/abs/2206.01256), which might be interesting future extensions to BEVFusion.


## Citation

If BEVFusion is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@inproceedings{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```
