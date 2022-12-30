"""Modification of the nuscenes dataset which incorporates tracking"""
import sys
sys.path.insert(0,'/btherien/github/nuscenes-devkit/python-sdk')

import tempfile
import nuscenes
import mmcv
import pyquaternion

import os.path as osp
import numpy as np
import os

from pyquaternion import Quaternion
from typing import Tuple, List, Dict
from mmdet.datasets import DATASETS
from ..core.bbox import LiDARInstance3DBoxes
from .nuscenes_dataset import NuScenesDataset, output_to_nusc_box, lidar_nusc_box_to_global
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval

class NuscenesBox_tracking(NuScenesBox):
    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 track: int = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        super().__init__(
                 center,
                 size,
                 orientation,
                 label,
                 score,
                 velocity,
                 name,
                 token)
        self.track = track
        

    def __eq__(self, other):
        se = super().__eq__(other)
        return se and self.track == other.track

    def __repr__(self):
        sr = super().__repr__()
        return sr + ", Trackid: {}".format(self.track)


def output_to_nusc_box_tracking(detection,tracking_classes_idx):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    tracks = detection['tracks_3d'].numpy()

    if len(box3d) == 0:
        return []

    try:
        box_gravity_center = box3d.gravity_center.numpy()
    except AttributeError:
        print('AttributeError')
        print(box3d)
        exit(0)
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        if labels[i] not in tracking_classes_idx:
            continue
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuscenesBox_tracking(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            track=tracks[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

@DATASETS.register_module()
class NuScenesDataset_tracking(NuScenesDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.nusc_obj = {}




    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_tracks = info['gt_tracks'][mask]
        gt_track_tte = info['gt_track_tte'][mask]

        try:
            gt_futures = info['gt_futures'][mask]
        except TypeError:
            print('TypeError')
            print('mask',mask)
            print('info[gt_futures]',info['gt_futures'],type(info['gt_futures']))
            # exit(0)
            gt_futures = np.array([])

        try:    
            gt_pasts = info['gt_pasts'][mask]
        except TypeError:
            print('TypeError')
            print('mask',mask)
            print('info[gt_pasts]',info['gt_pasts'],type(info['gt_pasts']))
            # exit(0)
            gt_pasts = np.array([])
        


        
        # print(info)
        # for k,v in info.items():
        #     print(k)
        # exit(0)

        # print('gt_pasts',gt_pasts.dtype,gt_pasts)
        # print('gt_futures',gt_futures.dtype,gt_futures)
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_tracks=gt_tracks,
            gt_track_tte=gt_track_tte,
            gt_futures=gt_futures,
            gt_pasts=gt_pasts)

        return anns_results


    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - track_fields (list): Track fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['track_tte_fields'] = []
        results['track_fields'] = []
        results['futures_fields'] = []
        results['pasts_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d




    def format_results_tracking(self, infos_idx, results, jsonfile_prefix=None, 
                                tracking_classes=['car', 'truck', 'bus', 'trailer',
                            'motorcycle', 'bicycle', 'pedestrian']):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            infos_idx (list[int]) : map from dataset order to infos index
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox_tracking(infos_idx, results, jsonfile_prefix, tracking_classes)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox_tracking(infos_idx, results_, tmp_file_, tracking_classes)})

        return result_files, tmp_dir


    def _format_bbox_tracking(self, infos_idx, results, jsonfile_prefix=None, 
                                tracking_classes=['car', 'truck', 'bus', 'trailer',
                            'motorcycle', 'bicycle', 'pedestrian']):
        """Convert the results to the standard format.

        Args:
            infos_idx (list[int]) : map from dataset order to infos index
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES
        idx_to_cls = {v:k for k,v in enumerate(mapped_class_names)}
        tracking_classes_idx = [idx_to_cls[x] for x in tracking_classes]

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            # print(det)
            sample_id = det['sample_idx']#infos_idx[sample_id]
            annos = []
            boxes = output_to_nusc_box_tracking(det,tracking_classes_idx)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                #[BEN COMMENT] this seems to be where they iterate over predicted
                #bounding boxes

                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.track,
                    attribute_name=attr)

                annos.append(nusc_anno)

            nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        
        return res_path
    

    def evaluate_tracking(self,
                 infos_idx,
                 results,
                 train=False,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 neptune=None):
        """Evaluation in nuScenes protocol for tracking method not yet defined 

            infos_idx (list[int]) : map from dataset order to infos index
        """
        result_files, tmp_dir = self.format_results_tracking(infos_idx, results, jsonfile_prefix)

        #TODO complete this methods with tracking visualizations

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                return self._evaluate_single_tracking(result_files[name],train,neptune=neptune)
            # results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            return self._evaluate_single_tracking(result_files,train,neptune=neptune)
        else:
            raise NotImplementedError('evaluate_tracking not implemented')


        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return results_dict


    def _evaluate_single_tracking(self,
                         result_path,
                         train=False,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox',
                         verbose=True,
                         quick=True,
                         neptune=None):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        import sys
        sys.path.insert(0,'/btherien/github/nuscenes-devkit/python-sdk')
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory

        output_dir = osp.join(*osp.split(result_path)[:-1])
        
        if train:
            eval_set_map = {
                'v1.0-mini': 'mini_train',
                'v1.0-trainval': 'train',
                'v1.0-medium': 'medium_train',
            }
        else:
            eval_set_map = {
                'v1.0-mini': 'mini_val',
                'v1.0-trainval': 'val',
                'v1.0-medium': 'medium_val',
            }
        
        if neptune:
            neptune['predicted_tracks_json'].upload(result_path)

        nusc_eval = TrackingEval(
                config=config_factory('tracking_nips_2019'), 
                result_path=result_path, 
                eval_set=eval_set_map[self.version], 
                output_dir=output_dir, 
                nusc_version=self.version, 
                nusc_dataroot=self.dataset_root, 
                verbose=verbose, 
                render_classes=None
            )

        #TODO complete this methods with proper results formatting
        metrics_summary = nusc_eval.main(render_curves=False)
        
        if not os.path.isdir('/btherien/tracking_results/'):
            os.mkdir('/btherien/tracking_results/')
        os.system("cp {} /btherien/tracking_results/metrics_summary{}.json".format(
            osp.join(output_dir, 'metrics_summary.json'),len(os.listdir('/btherien/tracking_results'))))

        # print(output_dir)
        # print(metrics_summary)
        return metrics_summary, nusc_eval
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']


        
        return detail, nusc_eval



    def _format_bbox_detection(self, infos_idx, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            infos_idx (list[int]) : map from dataset order to infos index
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            # sample_id = infos_idx[sample_id]
            sample_id = det['sample_idx']
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def get_nusc_obj(self,version,dataroot,verbose=False):
        nusc = self.nusc_obj.get(self.version,None)
        if nusc is None:
            nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
            self.nusc_obj[version] = nusc
        return nusc

    def _evaluate_single_detection(self,
                         result_path,
                         train=False,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = self.get_nusc_obj(version=self.version, dataroot=self.dataset_root, verbose=False)

        if train:
            eval_set_map = {
                'v1.0-mini': 'mini_train',
                'v1.0-trainval': 'train',
                'v1.0-medium': 'medium_train',
            }
        else:
            eval_set_map = {
                'v1.0-mini': 'mini_val',
                'v1.0-trainval': 'val',
                'v1.0-medium': 'medium_val',
            }

        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True)
        metrics_summary = nusc_eval.main(render_curves=False)

        return metrics_summary, nusc_eval

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results_detection(self, infos_idx, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox_detection(infos_idx, results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox_detection(infos_idx, results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate_detection(self,
                 infos_idx,
                 results,
                 train=False,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results_detection(infos_idx, results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                # ret_dict = 
                return self._evaluate_single_detection(result_files[name],train)
            # results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single_detection(result_files,train)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return results_dict



