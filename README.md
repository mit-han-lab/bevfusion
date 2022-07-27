# BEVFusion

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevfusion-multi-task-multi-sensor-fusion-with/3d-object-detection-on-nuscenes)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes?p=bevfusion-multi-task-multi-sensor-fusion-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevfusion-multi-task-multi-sensor-fusion-with/3d-multi-object-tracking-on-nuscenes)](https://paperswithcode.com/sota/3d-multi-object-tracking-on-nuscenes?p=bevfusion-multi-task-multi-sensor-fusion-with)

### [website](http://bevfusion.mit.edu/) | [paper](https://arxiv.org/abs/2205.13542) | [video](https://www.youtube.com/watch?v=uCAka90si9E)

![demo](assets/demo.gif)

## News

**If you are interested in getting updates, please sign up [here](https://docs.google.com/forms/d/e/1FAIpQLSfkmfsX45HstL5rUQlS7xJthhS3Z_Pm2NOVstlXUqgaK4DEfQ/viewform) to get notified!**

- **(2022/6/3)** BEVFusion ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions.
- **(2022/6/3)** We released the first version of BEVFusion (with pre-trained checkpoints and evaluation).
- **(2022/5/26)** BEVFusion is released on [arXiv](https://arxiv.org/abs/2205.13542).
- **(2022/5/2)** BEVFusion ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions that do not use test-time augmentation and model ensemble.

## Abstract

Multi-sensor fusion is essential for an accurate and reliable autonomous driving system. Recent approaches are based on point-level fusion: augmenting the LiDAR point cloud with camera features. However, the camera-to-LiDAR projection throws away the semantic density of camera features, hindering the effectiveness of such methods, especially for semantic-oriented tasks (such as 3D scene segmentation). In this paper, we break this deeply-rooted convention with BEVFusion, an efficient and generic multi-task multi-sensor fusion framework. It unifies multi-modal features in the shared bird's-eye view (BEV) representation space, which nicely preserves both geometric and semantic information. To achieve this, we diagnose and lift key efficiency bottlenecks in the view transformation with optimized BEV pooling, reducing latency by more than **40x**. BEVFusion is fundamentally task-agnostic and seamlessly supports different 3D perception tasks with almost no architectural changes. It establishes the new state of the art on the nuScenes benchmark, achieving **1.3%** higher mAP and NDS on 3D object detection and **13.6%** higher mIoU on BEV map segmentation, with **1.9x** lower computation cost.

## Results

### 3D Object Detection (on nuScenes test)

|   Model   | Modality | mAP  | NDS  |
| :-------: | :------: | :--: | :--: |
| BEVFusion-e |   C+L    | 74.99 | 76.09 |
| BEVFusion |   C+L    | 70.23 | 72.88 |

### 3D Object Detection (on nuScenes validation)

|        Model         | Modality | mAP  | NDS  | Checkpoint  |
| :------------------: | :------: | :--: | :--: | :---------: |
|    [BEVFusion](configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml)       |   C+L    | 68.39 | 71.32 | [Link](https://bevfusion.mit.edu/files/pretrained/bevfusion-det.pth) |
| [Camera-Only Baseline](configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml) |    C     | 33.25 | 40.15 | [Link](https://bevfusion.mit.edu/files/pretrained/camera-only-det.pth) |
| [LiDAR-Only Baseline](configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml)  |    L     | 64.68 | 69.28 | [Link](https://bevfusion.mit.edu/files/pretrained/lidar-only-det.pth) |

*Note*: The camera-only object detection baseline is a variant of BEVDet-Tiny with a much heavier view transformer and other differences in hyperparameters. Thanks to our [efficient BEV pooling](mmdet3d/ops/bev_pool) operator, this model runs fast and has higher mAP than BEVDet-Tiny under the same input resolution. Please refer to [BEVDet repo](https://github.com/HuangJunjie2017/BEVDet) for the original BEVDet-Tiny implementation. The LiDAR-only baseline is TransFusion-L.

### BEV Map Segmentation (on nuScenes validation)

|        Model         | Modality | mIoU | Checkpoint  |
| :------------------: | :------: | :--: | :---------: |
| [BEVFusion](configs/nuscenes/seg/fusion-bev256d2-lss.yaml)       |   C+L    | 62.69 | [Link](https://bevfusion.mit.edu/files/pretrained/bevfusion-seg.pth) |
| [Camera-Only Baseline](configs/nuscenes/seg/camera-bev256d2.yaml) |    C     | 56.56 | [Link](https://bevfusion.mit.edu/files/pretrained/camera-only-seg.pth) |
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
│   │   ├── nuscenes_infos_train_mono3d.coco.json
│   │   ├── nuscenes_infos_val_mono3d.coco.json
│   │   ├── nuscenes_infos_test_mono3d.coco.json
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

## FAQs

Q: Can we directly use the info files prepared by mmdetection3d?

A: We recommend re-generating the info files using this codebase since we forked mmdetection3d before their [coordinate system refactoring](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/changelog.md).

## Acknowledgements

BEVFusion is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [BEVDet](https://github.com/HuangJunjie2017/BEVDet), [TransFusion](https://github.com/XuyangBai/TransFusion), [CenterPoint](https://github.com/tianweiy/CenterPoint), [MVP](https://github.com/tianweiy/MVP), [FUTR3D](https://arxiv.org/abs/2203.10642), [CVT](https://github.com/bradyz/cross_view_transformers) and [DETR3D](https://github.com/WangYueFt/detr3d). 

Please also check out related papers in the camera-only 3D perception community such as [BEVDet4D](https://arxiv.org/abs/2203.17054), [BEVerse](https://arxiv.org/abs/2205.09743), [BEVFormer](https://arxiv.org/abs/2203.17270), [M2BEV](https://arxiv.org/abs/2204.05088), [PETR](https://arxiv.org/abs/2203.05625) and [PETRv2](https://arxiv.org/abs/2206.01256), which might be interesting future extensions to BEVFusion.


## Citation

If BEVFusion is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@article{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  journal={arXiv},
  year={2022}
}
```
