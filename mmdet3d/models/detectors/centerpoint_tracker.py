"""Module combines a 3D detector with a tracking module.

Author: Benjamin Therien
"""
import torch
import copy
import json
import time 

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from pyquaternion import Quaternion
from functools import reduce

from mmcv.runner import get_dist_info


from mmdet3d.models import build_model
from mmdet3d.models import FUSIONMODELS
# from mmdet3d.models.detectors.centerpoint_modif import loadPretrainedCenterpoint
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models import builder

from mmdet3d.core.bbox import LiDARInstance3DBoxes

from .bevfusion_modif import load_pretrained_detector

def confidence_score_filter(bbox_list,queries,threshold=0.01):
    if threshold == 0:
        return bbox_list,queries

    #filter out predictions with low confidence
    for i in range(len(bbox_list)):
        scores = bbox_list[i][1]
        mask = torch.where(scores > threshold)[0]

        print("\nconfidence_score_filter: keeping {}/{} predictions".format((scores > threshold).sum(),len(scores)))
        # print("\nscores:{}, mask:{}".format(scores.shape,mask.shape))
        bbox_list[i][0].tensor = bbox_list[i][0].tensor[mask,...] 
        bbox_list[i][1] = bbox_list[i][1][mask] 
        bbox_list[i][2] = bbox_list[i][2][mask]
        queries[i] = queries[i][mask,...]
        
    # print([[a.tensor.shape,b.shape,c.shape] for a,b,c in bbox_list])

    return bbox_list,queries

@FUSIONMODELS.register_module()
class CenterPointTracker(Base3DDetector):
    """Combining centerpoint with a tracking module."""

    def __init__(self,
                 net,
                 class_names, 
                 load_detector_from, 
                 pretrained_config,
                 bev_supervisor,
                 trk_manager,
                 confidence_threshold=0.01,
                 compute_loss_det=False,
                 test_output_config={'bbox':'track', 'score':'det'}, # 'gt' for ground truth
                 verbose=False,
                 use_neck=dict(use=False), 
                 use_backbone=dict(use=False),
                 use_middle_encoder=dict(use=False),
                 train_cfg=None, #for mmdet3d API
                 test_cfg=None, ):
        """
        Args:
            load_from (string): the filepath to the centerpoint checkpoint to be loaded
            pretrained_config (string): path to the pre-trained model's config file
            train_cfg (dict): for mmdet3d API
            test_cfg (dict): for mmdet3d API
            trk_manager (dict): arguments to the tracking module's constructor.
            compute_loss_det (bool): Whether to compute the loss of the detector or not.
        """
        super(CenterPointTracker, self).__init__()
        self.confidence_threshold = confidence_threshold
        pretrained_config = 'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
        load_detector_from = 'pretrained/bevfusion-det.pth'
        self.detector = load_pretrained_detector(config=pretrained_config,
                                                loadpath=load_detector_from,
                                                test=True)

        # print(self.detector)

        self.verbose = verbose
        self.use_neck = use_neck['use']
        self.use_backbone = use_backbone['use']
        self.use_middle_encoder = use_middle_encoder['use']
        self.compute_loss_det = compute_loss_det
        self.class_names = class_names
        self.test_output_config = test_output_config
        self.bev_supervisor = builder.build_supervisor(bev_supervisor)

        # ==================== load detector and build tracker ====================
        self.net = builder.build_tracker(net)
        self.trackManager = builder.build_tracker(trk_manager)
        # self.detector = loadPretrainedCenterpoint(config=pretrained_config,loadpath=load_detector_from,test=True)

        if self.use_neck:
            if use_neck['init'] == 'fine_tune':
                self.pts_neck = copy.deepcopy(self.detector.pts_neck)
            elif use_neck['init'] == 'new':
                self.pts_neck = builder.build_neck(use_neck['pts_neck'])
            else:
                raise ValueError("Invalid value for use_neck")
            
        if self.use_backbone:
            if use_backbone['init'] == 'fine_tune':
                self.pts_backbone = copy.deepcopy(self.detector.pts_neck)
            elif use_backbone['init'] == 'new':
                self.pts_backbone = builder.build_backbone(use_backbone['pts_backbone'])
            else:
                raise ValueError("Invalid value for use_backbone",use_backbone)

        if self.use_middle_encoder:
            if use_middle_encoder['init'] == 'fine_tune':
                self.pts_middle_encoder = copy.deepcopy(self.detector.pts_middle_encoder)
            elif use_middle_encoder['init'] == 'new':
                self.pts_middle_encoder = builder.build_middle_encoder(use_middle_encoder['pts_middle_encoder'])
            else:
                raise ValueError("Invalid value for use_backbone",use_backbone)


        # for n,p in self.detector.named_parameters():
        #     print(n)
        # exit(0)
        if compute_loss_det:
            #only freeze middle encoder
            for n,param in self.detector.named_parameters():
                if 'pts_middle_encoder' in n:
                    param.requires_grad = False
        else:
            #freeze everything
            for param in self.detector.parameters():
                param.requires_grad = False

        
        self.num_classes = len(self.class_names)
        self.cls_to_idx = {cls:i for i,cls in enumerate(self.class_names)}
        self.sample_to_ego_pose = load_nusc_ego_poses()
        self.sample_map = load_nusc_sample_transitions()
        self.rank,_ = get_dist_info()
        self.prev_sample = ''
        self.take_step = False
        self.forward_count = 0
        torch.autograd.set_detect_anomaly(True)

    def set_epoch(self,epoch,max_epoch):
        self.trackManager.set_epoch(epoch,max_epoch)
        self.epoch=epoch

    def validate_sequence_train(self, img_metas, first_in_scene):
        for meta, first in zip(img_metas,first_in_scene):
            if first:
                self.prev_sample = meta['token']
            else:
                try:
                    assert meta['token'] == self.sample_map[self.prev_sample]['next']
                    self.prev_sample = meta['token']
                except AssertionError:
                    print(meta['token'],"==",self.sample_map[self.prev_sample]['next'])
                    raise AssertionError

    


    def get_loss_train(self, losses_per_timestep, first_in_scene, last_in_scene, device):
        """Processes an arbitrary number of losses per timestep and returns a single loss at the right time."""

        losses = losses_per_timestep[0]
        #handle arbitrary number of losses and timesteps
        if len(losses_per_timestep) == 1:
            pass
        else:
            for loss_list in losses_per_timestep[1:]:
                assert len(loss_list) == len(losses)
                for i,loss in enumerate(loss_list):
                    losses[i] = losses[i] + loss


        output_loss = torch.tensor(0.,requires_grad=True,device=device)
        for i,loss in enumerate(losses):
            if first_in_scene[i]:
                try:
                    del self.track_loss_preserved
                    self.track_loss_preserved = losses[i]
                except AttributeError:
                    self.track_loss_preserved = losses[i]

            elif last_in_scene[i]:
                self.track_loss_preserved = self.track_loss_preserved + losses[i]
                output_loss = self.track_loss_preserved 
                self.take_step_now(True)
                self.track_loss_preserved = torch.tensor(0.,requires_grad=True,device=device)
            else:
                self.track_loss_preserved = self.track_loss_preserved + losses[i]

        
        if str(output_loss.clone().detach().cpu().item()) == 'nan':
            print("===========================================================")
            print("output_loss",output_loss)
            print("track_loss_preserved",self.track_loss_preserved)
            print("losses",losses)
            print("===========================================================")

        return output_loss

    def take_step_now(self,modify=False):
        if modify:#if modify then modify no matter what
            self.take_step = not self.take_step
            return not self.take_step
        else:#if not modify then only modify take step if it is true
            if self.take_step:
                self.take_step = not self.take_step
                return not self.take_step
            else:
                return self.take_step

    def get_ego_OandT(self,img_metas,first_in_scene):
        translations = []
        orientations = []
        for meta, first in zip(img_metas,first_in_scene):
            orientations.append(Quaternion(self.sample_to_ego_pose[meta['token']]['rotation']))#,dtype=np.float32))
            translations.append(
                np.array(self.sample_to_ego_pose[meta['token']]['translation'],dtype=np.float32))
        return translations, orientations

    def forward_bev(self, pts_feats, before_neck, before_backbone, middle_feats):
        if self.use_middle_encoder:
            before_backbone = self.pts_middle_encoder(*middle_feats)
        if self.use_backbone:
            before_neck = self.pts_backbone(before_backbone)
        if self.use_neck:
            pts_feats = self.pts_neck(before_neck)
        return pts_feats
        
    def set_train_mode(self):
        self.detector.eval()
        self.trackManager.train()
        self.net.train()
        if self.use_neck:
            self.pts_neck.train()
            set_bn_to_eval(self.pts_neck)
        if self.use_backbone:
            self.pts_backbone.train()
            set_bn_to_eval(self.pts_backbone)

    # 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_tracks', 'gt_futures', 'gt_pasts', 'gt_track_tte', 'camera_intrinsics', 'camera2ego', 'lidar2ego', 
    # 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix', 'metas', 'lidar2ego_translation', 'lidar2ego_rotation',
    #  'ego2global_rotation', 'ego2global_translation', 'first_in_scene', 'last_in_scene'

    
    def forward_train(self,
                      img,
                      points,
                      camera2ego,
                      lidar2ego,
                      lidar2camera,
                      lidar2image,
                      camera_intrinsics,
                      camera2lidar,
                      img_aug_matrix,
                      lidar_aug_matrix,
                      metas,
                      gt_masks_bev=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_tracks=None,
                      gt_futures=None,
                      gt_pasts=None,
                      gt_track_tte=None,
                      first_in_scene=None,
                      last_in_scene=None,
                      lidar2ego_translation=None,
                      lidar2ego_rotation=None,
                      ego2global_translation=None,
                      ego2global_rotation=None,
                      timestamp=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
            tracks (list[torch.tensor],optional): Ground truth tracking Identies
                for the current scene
            first_in_scene (list[bool],optional): Whether the corresponding 
                sample is the first in a scene
            last_in_scene (list[bool],optional): Whether the corresponding 
                sample is the last in the scene

        Returns:
             dict: Losses of different branches.
        """
        # print('in forward_train')
        #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchpack dist-run -np 8 python tools/train.py configs_tracking/centerpoint/centerpoint_decision_tracker_nus_20e_0-2_residual_focal_w65_g2_TF_all_bevfusion.py
        img_metas = metas
        for i in range(len(img_metas)):
            img_metas[i].update(dict(sample_idx=metas[i]['token']))
        
        # print(kwargs)
        # print(img_metas[0].keys())
        # print(img_metas[0])

        loss_dict = dict()
        if self.verbose:
            t1 = time.time()
            print("{} Starting forward_train()".format(self.log_msg()))

        losses_per_timestep = []
        log_vars = dict()
        device = next(self.parameters()).device

        self.set_train_mode() #put model in train mode

        self.validate_sequence_train(img_metas,first_in_scene)

        ego_translation, ego_orientation = self.get_ego_OandT(img_metas,first_in_scene)
        ego_update = {
            'ego2global_translation':ego2global_translation,
            'ego2global_rotation':ego2global_rotation,
            'lidar2ego_translation':lidar2ego_translation,
            'lidar2ego_rotation':lidar2ego_rotation,
            'ego_translation':ego_translation,
            'ego_orientation':ego_orientation,
            'timestamp':[x['timestamp'] for x in metas],
        }

        gt_tracks = [x.long().to(device) for x in gt_tracks]
        gt_bboxes_3d = [x.to(device) for x in gt_bboxes_3d]
        gt_labels_3d = [x.to(device) for x in gt_labels_3d]
        gt_track_tte = [x.long() for x in gt_track_tte]

        #TODO check why some futures are empty -- I think this happens when there are no detections in the frame
        # gt_futures = [torch.stack(x,dim=0) if x != [] else torch.zeros([],device=device) 
        #                 for x in gt_futures]
        # gt_pasts = [torch.stack(x,dim=0) if x != [] else torch.zeros([],device=device) 
        #                 for x in gt_pasts]

        #Forward pass through the detector
        losses, bbox_list, queries, pts_feats, before_neck, before_backbone, middle_feats = \
            self.detector.forward_detector_tracking(img,
                                                    points,
                                                    camera2ego,
                                                    lidar2ego,
                                                    lidar2camera,
                                                    lidar2image,
                                                    camera_intrinsics,
                                                    camera2lidar,
                                                    img_aug_matrix,
                                                    lidar_aug_matrix,
                                                    metas,
                                                    gt_masks_bev=None,
                                                    gt_bboxes_3d=None,
                                                    gt_labels_3d=None,
                                                    **kwargs)

        bbox_list = [[x['boxes_3d'],x['scores_3d'],x['labels_3d']] for x in bbox_list]
        queries = [x.squeeze(0).t() for x in queries]

        bbox_list, queries = confidence_score_filter(bbox_list,queries,threshold=self.confidence_threshold)

        


        
        if self.compute_loss_det:
            loss_dict.update(losses)


        out = self.forward_bev(pts_feats, before_neck, before_backbone, middle_feats)
        pts_feats = out #[torch.cat([x,y],dim=1) for x,y in zip(pts_feats,out)]

        if self.use_neck or self.use_backbone:
            #supervise BEV features 
            bev_losses, summary = self.bev_supervisor(forecast_net=self.net.forecast_bev,
                                                      pts_feats=pts_feats[0],
                                                      gt_labels=gt_labels_3d,
                                                      gt_bboxes=[x.tensor for x in gt_bboxes_3d],
                                                      gt_tracks=gt_tracks,
                                                      gt_futures=gt_futures,
                                                      first_in_scene=first_in_scene,
                                                      last_in_scene=last_in_scene,
                                                      device=device,
                                                      log_prefix='detBEV_')
            log_vars.update(summary)
            losses_per_timestep.append(bev_losses)
        

        #forward pass through the tracker
        tracking_losses, log_vars_update, _ = self.trackManager.addDetection(
                net=self.net,
                points=points,
                queries=queries,
                pts_feats=pts_feats, 
                bbox_list=bbox_list, 
                img_metas=img_metas,
                first_in_scene=first_in_scene,
                last_in_scene=last_in_scene,
                ego_update=ego_update,
                output_preds=False,
                gt_labels=gt_labels_3d, 
                gt_bboxes=gt_bboxes_3d, 
                gt_tracks=gt_tracks,
                gt_futures=gt_futures,
                gt_pasts=gt_pasts,
                gt_track_tte=gt_track_tte,
                device=device,
            )
        losses_per_timestep.append(tracking_losses)

        self.forward_count += 1
        log_vars.update(log_vars_update)
        
        output_loss = self.get_loss_train(losses_per_timestep,first_in_scene,last_in_scene,device)
        loss_dict.update({'tracking_loss': output_loss})
        
        if self.verbose:
            print("{} Ending forward forward_train after {}s".format(self.log_msg(),self.rank,time.time()-t1))

        return loss_dict, log_vars 


    
    def validate_sequence_test(self, img_metas, padded):
        for meta,pad in zip(img_metas,padded):
            if pad:
                continue

            if self.sample_map[meta['token']]['prev'] == '':
                self.prev_sample = meta['token']
            else:
                try:
                    assert meta['token'] == self.sample_map[self.prev_sample]['next']
                    self.prev_sample = meta['token']
                except AssertionError:
                    print(meta['token'],"==",self.sample_map[self.prev_sample]['next'])
                    raise AssertionError


    def set_eval_mode(self):
        self.detector.eval()
        self.net.eval()
        self.trackManager.eval()
        if self.use_neck:
            self.pts_neck.eval()
        if self.use_backbone:
            self.pts_backbone.eval()


    def simple_test(self,
                      img,
                      points,
                      camera2ego,
                      lidar2ego,
                      lidar2camera,
                      lidar2image,
                      camera_intrinsics,
                      camera2lidar,
                      img_aug_matrix,
                      lidar_aug_matrix,
                      metas,
                      gt_masks_bev=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_tracks=None,
                      gt_futures=None,
                      gt_pasts=None,
                      gt_track_tte=None,
                      first_in_scene=None,
                      last_in_scene=None,
                      lidar2ego_translation=None,
                      lidar2ego_rotation=None,
                      ego2global_translation=None,
                      ego2global_rotation=None,
                      timestamp=None,
                      padded=None,
                      **kwargs):
        """Test function without augmentaiton for tracking."""
        img_metas = metas
        for i in range(len(img_metas)):
            img_metas[i].update(dict(sample_idx=metas[i]['token']))
        # print('in simple test')
        self.set_eval_mode()
        device = next(self.parameters()).device

        if type(img_metas) != list:
            img_metas = [img_metas]

        if type(points) != list:
            points = [points]

        self.validate_sequence_test(img_metas,padded)
        

        ego_translation, ego_orientation = self.get_ego_OandT(img_metas,first_in_scene)
        ego_update = {
            'ego2global_translation':ego2global_translation,
            'ego2global_rotation':ego2global_rotation,
            'lidar2ego_translation':lidar2ego_translation,
            'lidar2ego_rotation':lidar2ego_rotation,
            'ego_translation':ego_translation,
            'ego_orientation':ego_orientation,
            'timestamp':[x['timestamp'] for x in metas],
        }

        gt_tracks = [x.long().to(device) for x in gt_tracks]
        gt_bboxes_3d = [x.to(device) for x in gt_bboxes_3d]
        gt_labels_3d = [x.to(device) for x in gt_labels_3d]
        gt_track_tte = [x.long() for x in gt_track_tte]

        #Forward pass through the detector
        losses, bbox_list, queries, pts_feats, before_neck, before_backbone, middle_feats = \
            self.detector.forward_detector_tracking(img,
                                                    points,
                                                    camera2ego,
                                                    lidar2ego,
                                                    lidar2camera,
                                                    lidar2image,
                                                    camera_intrinsics,
                                                    camera2lidar,
                                                    img_aug_matrix,
                                                    lidar_aug_matrix,
                                                    metas,
                                                    gt_masks_bev=None,
                                                    gt_bboxes_3d=None,
                                                    gt_labels_3d=None,
                                                    **kwargs)
        bbox_list = [[x['boxes_3d'],x['scores_3d'],x['labels_3d']] for x in bbox_list]
        queries = [x.squeeze(0).t() for x in queries]

        bbox_list, queries = confidence_score_filter(bbox_list,queries,threshold=self.confidence_threshold)

        # print(bbox_list[0][0])
        # exit(0)

        pts_feats = self.forward_bev(pts_feats, before_neck, before_backbone, middle_feats)


        # print(type(middle_feats))
        # print('middle_feats',[x.shape for x in middle_feats] )
        # print('before_backbone', [x.shape for x in before_backbone ] )
        # print('before_neck',[x.shape for x in before_neck])
            
        tracking_losses, log_vars_update, tracking_preds, tracking_score, tracking_bbox, tracking_labels = \
            self.trackManager.addDetection_test(net=self.net,
                                                points=points,
                                                queries=queries,
                                                pts_feats=pts_feats, 
                                                bbox_list=bbox_list, 
                                                img_metas=img_metas,
                                                first_in_scene=first_in_scene, 
                                                last_in_scene=last_in_scene,
                                                ego_update=ego_update,
                                                gt_labels=gt_labels_3d, 
                                                gt_bboxes=gt_bboxes_3d, 
                                                gt_tracks=gt_tracks,
                                                gt_futures=gt_futures,
                                                gt_pasts=gt_pasts,
                                                gt_track_tte=gt_track_tte,
                                                padded=padded,
                                                device=device,)
        
        if self.test_output_config == 'gt':
            #TESTING WITH THE GROUND TRUTH
            bbox_pts = self.get_gt_results(gt_bboxes_3d,gt_tracks,gt_labels_3d,device)
        else:
            bbox_pts = self.get_predicted_results(tracking_labels,
                                                  tracking_preds,
                                                  tracking_score,
                                                  tracking_bbox,
                                                  bbox_list,
                                                  img_metas)

        bbox_out = [dict() for i in range(len(img_metas))]

        for result_dict, pts_bbox in zip(bbox_out, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        for i in range(len(bbox_out)):
            if padded[i]:
                bbox_out[i] = {'pts_bbox' : False}
            else:
                bbox_out[i]['pts_bbox']['token'] = img_metas[i]['token']


        

        return bbox_out, [log_vars_update]
    
    

    

    def get_predicted_results(self,tracking_labels,tracking_preds,tracking_score,tracking_bbox,bbox_list,img_metas):
        """get results from detection scores and bboxes"""
        #TESTING WITH PREDICTIONS
        for i in range(len(tracking_preds)):
            trk_idx = torch.where(reduce(torch.logical_or,
                                     [tracking_labels[i] == self.cls_to_idx[x] 
                                      for x in self.trackManager.tracked_classes]))

            
            tracking_labels[i] = tracking_labels[i][trk_idx] #get track labels
            tracking_preds[i] = tracking_preds[i][trk_idx]   #get track indices
            tracking_score[i] = tracking_score[i][trk_idx]
            tracking_bbox[i] = tracking_bbox[i][trk_idx]


            if tracking_bbox[i].nelement() != 0:
                try:
                    tracking_bbox[i] = LiDARInstance3DBoxes(tracking_bbox[i],tracking_bbox[i].shape[1]) # img_metas[i]['box_type_3d'](tracking_bbox[i],tracking_bbox[i].shape[1])
                except IndexError:
                    print("[Error in Evaluate CenterPointTracker get_predicted_results()] ############################################")
                    print('tracking_bbox',tracking_bbox)
                    print('tracking_bbox[i]',tracking_bbox[i])
                    print('tracking_bbox[i].shape',tracking_bbox[i].shape)
                    exit(0)
                    pass
                         
        try:
            assert tracking_preds[0].unique().shape[0] == tracking_preds[0].shape[0], "ERROR duplicate tracking id in output tracks."
        except AssertionError as e:
            print("AssertionError in Evaluate CenterPointTracker get_predicted_results()] ############################################")
            print(e)
            print('tracking_preds[0].unique()',tracking_preds[0].unique())
            print('tracking_preds[0]',tracking_preds[0])
            exit(0)


        bbox_pts = [
            bbox3d2result_tracking(bboxes, scores, labels, tracks)
            for bboxes, scores, labels, tracks in zip(tracking_bbox,tracking_score,tracking_labels,tracking_preds)
        ]
        return bbox_pts


    def get_gt_results(self,gt_bboxes_3d,gt_tracks,gt_labels_3d,device):
        """Retrieves the results produced by the ground truth annotations (for testing puposes)"""
        gt_bboxes_3d = [x.to(device) for x in gt_bboxes_3d]
        gt_tracks = [x.to(device) for x in gt_tracks]
        gt_labels_3d = [x.to(device) for x in gt_labels_3d]
        for i in range(len(gt_bboxes_3d)):
            idx = torch.where(reduce(torch.logical_or,
                                     [gt_labels_3d[i] == self.cls_to_idx[x] 
                                      for x in self.trackManager.tracked_classes]))

            gt_bboxes_3d[i] = gt_bboxes_3d[i][idx]
            gt_tracks[i] = gt_tracks[i][idx]
            gt_labels_3d[i] = gt_labels_3d[i][idx]

        bbox_out = [dict() for i in range(len(gt_bboxes_3d))]
        gt_scores = [torch.ones(x.tensor.size(0),device=device) for x in gt_bboxes_3d]
        bbox_pts = [
            bbox3d2result_tracking(bboxes, scores, labels, tracks)
            for bboxes, scores, labels, tracks in zip(gt_bboxes_3d,gt_scores,gt_labels_3d,gt_tracks)
        ]
        return bbox_out, bbox_pts
    


    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        
        if self.verbose:
            t1 = time.time()
            print("{} Starting train_step()".format(self.log_msg()))

        # print('in train_step()')
        # print(data.keys())
        # exit(0)

        losses, log_vars_train = self(**data)
        loss, log_vars = self._parse_losses(losses)

        log_vars.update(log_vars_train)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['metas']))
        
        if self.verbose:
            print("{} Ending train_step() after {}s".format(self.log_msg(),time.time()-t1))

        return outputs

    def forward_test(self, *args, **kwargs):
        """Copied from Base3DDetector

        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        return self.simple_test( *args, **kwargs)
        # print('forward_test metas',img_metas)

        for var, name in [(points, 'points'), (metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(metas)))

        # print(kwargs)
        # print(kwargs.keys())
        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, metas, img, **kwargs)


    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['metas']))

        return outputs

    # ====================================== Printing ======================================

    def log_msg(self):
        return"[CenterPointTracker | Rank:{}, forward_count:{}]".format(self.rank,self.forward_count)
        
    def display_parameters(self):
        before_second_count = 0
        pts_backbone_count = 0
        pts_neck_count = 0
        pts_bbox_head_count = 0
        for name, param in self.detector.named_parameters():
            if 'pts_backbone' in name:
                pts_backbone_count += param.numel()
            elif 'pts_neck' in name:
                pts_neck_count += param.numel()
            elif 'pts_bbox_head' in name:
                pts_bbox_head_count += param.numel()
            else:
                before_second_count += param.numel()

        total_cpoint_params = sum(p.numel() for p in self.detector.parameters())
        print('####################################################################################')    
        print('\n\nCenterpoint Parameters: {:.4f}M'.format(total_cpoint_params/1000000)) 
        print('Centerpoint before SECOND Parameters: {:.4f}M'.format(before_second_count/1000000))
        print('Centerpoint SECOND Parameters: {:.4f}M'.format(pts_backbone_count/1000000))
        print('Centerpoint FPN Parameters: {:.4f}M'.format(pts_neck_count/1000000))
        print('Centerpoint head Parameters: {:.4f}M'.format(pts_bbox_head_count/1000000))
        print('Centerpoint + tracker Parameters: {:.4f}M'.format(sum(p.numel() for p in self.parameters())/1000000))
        print('\n\n')
        print('####################################################################################')

    def visualizeBEVPreds(self,pts_bbox_outs,preds,batch=0,plots_per_row=2):
        """Plots centerpoint predictions for each class in the BEV map.

        Args:
            pts_bbox_outs (list[misc]): list of diffrent predictions from 
                centerpoints detection head.
            preds (list[misc]): list of diffrent predictions from 
                centerpoint.
            batch (int): index in the batch to process.
            plots_per_row (int): number of plots to display in each row.
        """
        #TODO configure the hetmap axes in order to have them matchup to the scene limits
        #TODO remove batch dependence
        idx_to_bev = {}
        idx_to_cls = {}
        for i,x in enumerate(self.class_names):
            for ii,cls in enumerate(x):
                idx_to_cls[len(idx_to_cls)] = cls
                idx_to_bev[len(idx_to_bev)] = pts_bbox_outs[i][0]['heatmap'][batch,ii,:,:]
                
                
        n = np.ceil(len(idx_to_cls)/plots_per_row).astype(np.int32)
        fig, axs = plt.subplots(n, plots_per_row, figsize=(10,20))
        for x in range(len(idx_to_bev)):
            c,r = x % plots_per_row, int(x/plots_per_row)
            axs[r,c].set_title("Class: {}".format(idx_to_cls[x]))
            axs[r,c].imshow(idx_to_bev[x].detach().cpu().numpy().T,
                        origin="lower", cmap='jet', interpolation='nearest')
            
            coors = preds[batch][3][torch.where(preds[batch][2]==x)]
            print(coors.shape)
            
            
            axs[r,c].scatter(
                coors[:,1].detach().cpu().numpy().astype(np.int32), 
                coors[:,0].detach().cpu().numpy().astype(np.int32), 
                
                s=80, c=None, facecolors='none', edgecolors='y'
            )
            
        fig.show()


    # ====================================== For MMdet3d API ======================================

    def init_weights(self):
        """Initialize the weights."""
        return None

    def extract_feat(self,*args,**kwargs):#abstract method needs to be re-implemented
        return self.detector.extract_feat(*args,**kwargs)

    def show_result(self):
        return self.detector.show_results()

    def aug_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def simple_test_pts(self, bbox_list, tracking_list, img_metas):
        """Test function of point cloud branch."""
        print("Warning, entering recently deprecated function: simple_test_pts")
        pass
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels, tracks)
        #     for ((bboxes, scores, labels), tracks) in zip(bbox_list, tracking_list)
        # ]
        # return bbox_results



# ====================================== Helpers ======================================



def set_bn_to_eval(a):
    for x in a.children():
        if len([k for k in x.children()]) > 0:
            set_bn_to_eval(x)
        elif isinstance(x,nn.BatchNorm2d):
            x.eval()
        else:
            pass 

def load_nusc_ego_poses(version='v1.0-trainval',data_root='data/nuscenes/'):
    """Loads a cached version of the scene map or creates it outright"""
    filepath = osp.join(data_root,"scene_to_ego_pose_{}.json".format(version))
    try:
        sampleMap = json.load(open(filepath,'r'))
    except Exception:
        from nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        sampleMap = {}
        for sample in nusc.sample:
            sample_lidar_top = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
            sample_ego_pose = nusc.get('ego_pose',sample_lidar_top['ego_pose_token'])
            sampleMap[sample['token']] = {
                "prev":sample['prev'],
                "next":sample['next'],
                "rotation":sample_ego_pose['rotation'],
                "translation":sample_ego_pose['translation']
            }
                
        json.dump(sampleMap, open(filepath,'w'))
    return sampleMap

def load_nusc_sample_transitions(version='v1.0-trainval',data_root='data/nuscenes/'):
    """Loads a cached version of the scene map or creates it outright"""
    filepath = osp.join(data_root,"scene_transitions_{}.json".format(version))
    try:
        sampleMap = json.load(open(filepath,'r'))
    except Exception:
        from nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        sampleMap = {}
        print(nusc.__dict__.keys())
        for sample in nusc.sample:
            sampleMap[sample['token']] = {
                "prev":sample['prev'],
                "next":sample['next']
            }

        json.dump(sampleMap, open(filepath,'w'))
    return sampleMap


def bbox3d2result_tracking(bboxes, scores, labels, track_predictions, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        track_predictions (torch.tensor): Track idx if shape (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - tracks_3d (torch.Tensor): track ids.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
        tracks_3d=track_predictions.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict

