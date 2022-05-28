# BEVFusion

### [website](http://bevfusion.mit.edu/) | [paper](https://arxiv.org/abs/2205.13542) | [video](https://www.youtube.com/watch?v=uCAka90si9E)

![demo](assets/demo.gif)

## News

**If you are interested in getting updates, please sign up [here](https://docs.google.com/forms/d/e/1FAIpQLSfkmfsX45HstL5rUQlS7xJthhS3Z_Pm2NOVstlXUqgaK4DEfQ/viewform) to get notified!**

- **(2022/5/28)** We will be releasing the first version of BEVFusion next week. Stay tuned!

## Abstract

Multi-sensor fusion is essential for an accurate and reliable autonomous driving system. Recent approaches are based on point-level fusion: augmenting the LiDAR point cloud with camera features. However, the camera-to-LiDAR projection throws away the semantic density of camera features, hindering the effectiveness of such methods, especially for semantic-oriented tasks (such as 3D scene segmentation). In this paper, we break this deeply-rooted convention with BEVFusion, an efficient and generic multi-task multi-sensor fusion framework. It unifies multi-modal features in the shared bird's-eye view (BEV) representation space, which nicely preserves both geometric and semantic information. To achieve this, we diagnose and lift key efficiency bottlenecks in the view transformation with optimized BEV pooling, reducing latency by more than **40x**. BEVFusion is fundamentally task-agnostic and seamlessly supports different 3D perception tasks with almost no architectural changes. It establishes the new state of the art on the nuScenes benchmark, achieving **1.3%** higher mAP and NDS on 3D object detection and **13.6%** higher mIoU on BEV map segmentation, with **1.9x** lower computation cost.

## Results

### 3D Object Detection on nuScenes test

|   Model   | Modality | mAP  | NDS  | Checkpoint  |
| :-------: | :------: | :--: | :--: | :---------: |
| BEVFusion |   C+L    | 70.2 | 72.9 | Coming Soon |

### 3D Object Detection on nuScenes validation

|        Model         | Modality | mAP  | NDS  | Checkpoint  |
| :------------------: | :------: | :--: | :--: | :---------: |
|      BEVFusion       |   C+L    | 68.5 | 71.4 | Coming Soon |
| Camera-Only Baseline |    C     | 33.3 | 40.2 | Coming Soon |
| LiDAR-Only Baseline  |    L     | 64.8 | 69.3 | Coming Soon |

### BEV Map Segmentation on nuScenes validation

|        Model         | Modality | mIoU | Checkpoint  |
| :------------------: | :------: | :--: | :---------: |
|      BEVFusion       |   C+L    | 62.7 | Coming Soon |
| Camera-Only Baseline |    C     | 56.6 | Coming Soon |
| LiDAR-Only Baseline  |    L     | 48.6 | Coming Soon |

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
