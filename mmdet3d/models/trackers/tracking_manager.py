""" PnP-Net style tracking module for centerpoint 

Author: Benjamin Therien
"""
import gc
import torch
import copy
import time

import numpy as np
import torch.nn as nn
from .transforms import affine_transform, apply_transform

from .tracking_helpers import update_log_vars, getParamInfo
from torch.nn.utils.rnn import pad_sequence
from mmdet3d.models import TRACKERS
from mmdet3d.models  import builder
from pyquaternion import Quaternion

from mmcv.runner import get_dist_info
from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
from mmdet.core.bbox.iou_calculators.builder import  build_iou_calculator
from mmdet3d.models.builder import SUPERVISORS, build_supervisor
from .tracking_helpers import get_bbox_sides_and_center, EmptyBBox

from functools import reduce




class EgoVehicle:

    def init_attributes(self):
        self.position = None
        self.orientation = None
        self.velocity = None
        self.yaw_feat = None
        self.yaw = None
        self.o_velocity = None
        self.l2et = None
        self.l2er = None

    def __init__(self):
        super().__init__()
        self.init_attributes()

    def reset(self,net,device):
        """reset the ego"""
        del self.position
        del self.orientation
        del self.velocity
        del self.yaw
        del self.yaw_feat
        del self.o_velocity
        del self.l2et
        del self.l2er
        self.init_attributes()


    def add_timestep(self,ego_orientation,ego_translation,lidar2ego_translation,lidar2ego_rotation,
                     ego2global_translation,ego2global_rotation,device):
        """ Adds the orientation and position of ego for the current timestep

        args:
            position [np.array] : current position of the ego
            orientation [Quaternion] : current orientation of the ego
        """
        if self.position == None: 
            self.orientation = [ego_orientation]
            self.yaw = [torch.tensor(quaternion_yaw(ego_orientation),dtype=torch.float32,device=device)]
            self.yaw_feat = [torch.tensor([torch.sin(self.yaw[-1]),torch.cos(self.yaw[-1])],dtype=torch.float32,device=device)]
            self.position = [torch.tensor(ego_translation,dtype=torch.float32,device=device)]
            self.velocity = [torch.tensor([0.,0.],dtype=torch.float32,device=device)]
            self.o_velocity = [torch.tensor([np.sin(0.),np.cos(0.)],dtype=torch.float32,device=device)]
            self.l2et = [lidar2ego_translation.cpu().numpy()]
            self.l2er = [lidar2ego_rotation.cpu().numpy()]
            self.e2gt = [ego2global_translation.cpu().numpy()]
            self.e2gr = [ego2global_rotation.cpu().numpy()]
        else:
            self.orientation.append(ego_orientation)
            self.yaw.append(torch.tensor(quaternion_yaw(ego_orientation),dtype=torch.float32,device=device))
            self.yaw_feat.append(torch.tensor([torch.sin(self.yaw[-1]),torch.cos(self.yaw[-1])],dtype=torch.float32,device=device))
            self.position.append(torch.tensor(ego_translation,dtype=torch.float32,device=device))
            self.velocity.append(
                (self.position[-2][:2] - self.position[-1][:2]).abs() / 0.5
            )
            angle = angle_diff(self.yaw[-1],self.yaw[-2],period=2*np.pi) / 0.5
            self.o_velocity.append(
                torch.tensor([torch.sin(angle),torch.cos(angle)],dtype=torch.float32,device=device)
            )
            self.l2et.append(lidar2ego_translation.cpu().numpy())
            # self.l2er.append(Quaternion(lidar2ego_rotation.cpu().numpy()))
            self.l2er.append(lidar2ego_rotation.cpu().numpy())
            self.e2gt.append(ego2global_translation.cpu().numpy())
            # self.e2gr.append(Quaternion(ego2global_rotation.cpu().numpy()))
            self.e2gr.append(ego2global_rotation.cpu().numpy())


    def transform_over_time(self,points,from_=None,to_=None):
        """Transforms positions from the egos coordinate frame at 
            from_ to positions in its coordinate frame at to_.

            args:
                points [torch.tensor] : (N,3) xy points in the lidar coordinate frame 
                from_ [int] : timestep to transform from
                to_ [int] : timestep to transform to
        """
        if from_ == None and to_ == None: 
            from_ = len(self.position) - 2 #prev
            to_ = len(self.position) - 1 #current

        if max(from_,to_) > len(self.position):
            raise ValueError("Tried to transform using a non-existant timestep")

        #add secret sauce
        lidar2global = affine_transform(
            rotation=np.roll(self.e2gr[from_],-1),
            rotation_format='quat',
            translation=self.e2gt[from_],
        ) @ affine_transform(
            rotation=np.roll(self.l2er[from_],-1),
            rotation_format='quat',
            translation=self.l2et[from_],
        )

        #add secret sauce
        global2lidar = np.linalg.inv(affine_transform(
            rotation=np.roll(self.l2er[to_],-1),
            rotation_format='quat',
            translation=self.l2et[to_],
        )) @ np.linalg.inv(affine_transform(
            rotation=np.roll(self.e2gr[to_],-1),
            rotation_format='quat',
            translation=self.e2gt[to_],
        ))

        return apply_transform(global2lidar @ lidar2global, points)


    def get_current_feat(self):
        """Returns the features of the ego for the current timestep"""
        return torch.cat([self.velocity[-1],
                          self.o_velocity[-1],
                          self.yaw_feat[-1]]).unsqueeze(0)

        
def non_max_suppression(bboxes, confidence_scores, suppress_threshold, ioucal, device, pred_labels=None):
    iou = ioucal(bboxes,bboxes)

    if pred_labels != None:
        mask = torch.zeros((len(pred_labels),len(pred_labels)),dtype=torch.float32, device=device)
        cp = torch.cartesian_prod(torch.arange(0,len(pred_labels)), torch.arange(0,len(pred_labels))).to(device)
        mask[cp[:,0],cp[:,1]] = (pred_labels[cp[:,0]] != pred_labels[cp[:,1]]).float() * -10000.0
        iou = iou + mask

    iou = torch.triu(iou, diagonal=1) # suppress duplicate entires 
    idx1,idx2 = torch.where(iou > suppress_threshold)

    scores = confidence_scores[idx1] - confidence_scores[idx2]
    suppress = torch.cat([idx1[torch.where(scores <= 0)], idx2[torch.where(scores > 0)]])


    idx = torch.arange(0,bboxes.size(0))
    idx[suppress] = -1
    keep = torch.where(idx >= 0)[0]

    return keep, suppress


@TRACKERS.register_module()
class TrackingManager(nn.Module):
    """ TrackingManager is a tracking module which follows the tracking
    by detection paradigm. Therefore, it can be used with any 3D detector.
    """
    def __init__(self,
                 tracker,
                 cls_to_idx,
                 point_cloud_range,
                 det_nms_threshold,
                 det_nms_ioucal,
                 is_test_run=False,
                 use_det_nms=True,
                 bev_feat_supervisor=None,
                 verbose=False,
                 bev_gather_mode='interpolate',
                 tracked_classes=['car', 'truck', 'bus', 'trailer',
                            'motorcycle', 'bicycle', 'pedestrian'], # from the nuscenes challenge
                 tracked_range={'car':50, 'truck':50, 'bus':50, 'trailer':50,
                                    'motorcycle':40, 'bicycle':40, 'pedestrian':40} ):
        """Complete Tracking Module
    
        Params:
            cls_to_idx (dict, required): Mapping from classes to 
                their IDs
            bev_dim (int, required): the dimension of BEV features 
            lstm_hidden_dim (int, required): the hidden dimension of the LSTM
            lstm_in_dim (int, required): input dimension of the LSTM
                detection and track pairs. tracks are only matched if their score is small enough
            tracked_classes (list[string]): The different classes which 
                should be tracked by our module 
            tracked_range (dict): Map between class name string and the maximal
                range cutoff for tracking
        """ 
        #TODO make the different parameters class specific, allowing 
        # for different track thresholds for different classes 
        super().__init__()
        self.tracked_classes = tracked_classes
        self.cls_to_idx = cls_to_idx
        self.det_nms_threshold = det_nms_threshold
        self.det_nms_ioucal = build_iou_calculator(det_nms_ioucal)
        self.use_det_nms = use_det_nms
        self.is_test_run = is_test_run

        if bev_feat_supervisor:
            self.bev_feat_supervisor = build_supervisor(bev_feat_supervisor)
        else:
            self.bev_feat_supervisor = None
            
        rank, world_size = get_dist_info()
        self.count_reset = rank * 100000
        self.rank = rank

        self.trackers = {}

        # for k in self.cls_to_idx.keys():
        #     if k in self.tracked_classes:
        #         temp = copy.deepcopy(tracker)
        #         temp.update(dict(cls=k,rank=self.rank))
        #         self.trackers[k] = builder.build_tracker(temp)
        tracker.update(dict(rank=self.rank))
        self.tracker = builder.build_tracker(tracker)

        self.ego = EgoVehicle()
        self.tracked_range = tracked_range
        self.timestep = 0
        self.trackCount = 0
        # self.confidence_weight = torch.nn.Parameter(torch.tensor([1.],dtype=torch.float32))
        
        self.gradMonitor = None
        self.paramMonitor = None
        
        self.verbose = verbose
        self.point_cloud_range = point_cloud_range

        self.iter_count = 0
        self.bev_gather_mode = bev_gather_mode


    def log_msg(self):
        return "[TrackingManager|Rank:{}, Timestep:{}, trackCount:{}]".format(self.rank,self.timestep,self.trackCount)


    def set_epoch(self,epoch,max_epoch):
        self.tracker.set_epoch(epoch,max_epoch)
        # for k in self.trackers.keys():
        #     self.trackers[k].set_epoch(epoch,max_epoch)


    def reset(self,net,device):
        """reset the tracker after a sequence ends"""
        # sif self.verbose:
        # print('[TrackingManager - info] Resetting for new scene, previous scene lasted {} timesteps.'.format(self.timestep+1))
        # for v in self.trackers.vaslues():
            # v.reset(net,device)
        self.tracker.reset(net,device)
        self.timestep = -1
        if self.training:
            self.trackCount = self.count_reset 
        self.ego.reset(net,device)
        torch.cuda.empty_cache() 
        gc.collect()

    
    
    
    def addDetection(self,net,points,queries,pts_feats,bbox_list,img_metas,first_in_scene,last_in_scene,ego_update,
                     gt_labels,gt_bboxes,gt_tracks,device,gt_futures, gt_pasts,gt_track_tte,
                     output_preds=False,return_loss=True,
                     ):
        """ Adds the detections from the current timestep.
        
        Args
            pts_feats (torch.Tensor): BEVMap features
            bbox_list (torch.Tensor): label prediction for each 
                detection
            img_metas (list[dict]): Meta information of each sample.
                Defaults to None
            first_in_scene (list[bool]): Whether the corresponding 
                sample is the first in a scene
            gt_labels (torch.Tensor): GT class labels
            gt_bboxes (LiDARInstance3DBoxes): GT bounding boxes
            gt_tracks (torch.Tensor): GT tracks ids
            output_preds (bool): If True outputs tracking predictions o/w None
            return_loss (bool): If True computes the tracking loss o/w None

        Returns:
            tracking_losses (dict): losses computed from the association matrix
                for each class and each sample
        """
        if self.verbose:
            t1 = time.time()
            print("{} Entering addDetection".format(self.log_msg()))


        if self.is_test_run:
            exit(0)

        all_track_preds = [] if output_preds else None
        all_track_score = [] if output_preds else None
        all_track_bbox = [] if output_preds else None
        all_track_labels = [] if output_preds else None
        

        batch,f,h,w = pts_feats[0].shape
        log_vars = {}
        tracking_losses = []

        for i in range(batch):
            if first_in_scene[i]:
                self.reset(net,device)

            self.timestep += 1

            self.ego.add_timestep(
                device=device,
                **{k:x[i]for k,x in ego_update.items()}
            )
            
            if bbox_list[i][1].size(0) == 0:
                merged = torch.tensor([],device=device)
            else:
                merged, bev_feats, motion_feats = net.getMergedFeats(
                    ego=self.ego,
                    pred_cls=bbox_list[i][2],
                    bbox_feats=bbox_list[i][0].tensor.to(device),
                    queries=queries[i],
                    bbox_side_and_center=get_bbox_sides_and_center(bbox_list[i][0]),
                    bev_feats=pts_feats[0][i,...],
                    confidence_scores=bbox_list[i][1].to(device),
                    point_cloud_range=self.point_cloud_range,
                    device=device,
                )

                # get loss for merged feat supervision
                if return_loss and self.bev_feat_supervisor != None:
                    loss, summary, log = self.bev_feat_supervisor(
                        net=net,
                        bev_feats=bev_feats,
                        pred_bboxes=bbox_list[i][0],
                        gt_bboxes=gt_bboxes[i],
                        gt_futures=gt_futures[i],
                        device=device
                    )
                    log_vars.update(summary)

                    try:
                        tracking_losses[i] = tracking_losses[i] + loss
                    except IndexError: 
                        tracking_losses.append(loss)


            if output_preds:
                batch_track_preds = []
                batch_track_score = []
                batch_track_bbox = []
                batch_track_labels = []


            # cls_idx = self.cls_to_idx[cls]
            pred_idx = torch.where(reduce(torch.logical_or,
                                          [bbox_list[i][2] == self.cls_to_idx[cls_] for cls_ in self.tracked_classes]))[0]
            gt_idx = torch.where(reduce(torch.logical_or,
                                        [gt_labels[i] == self.cls_to_idx[cls_] for cls_ in self.tracked_classes]))[0]

            if len(pred_idx) > 0 and self.use_det_nms:
                keep, suppress =  non_max_suppression(bboxes=bbox_list[i][0][pred_idx].tensor, 
                                                      confidence_scores=bbox_list[i][1][pred_idx], 
                                                      suppress_threshold=self.det_nms_threshold, 
                                                      ioucal=self.det_nms_ioucal,
                                                      device=device,
                                                      pred_labels=bbox_list[i][2][pred_idx])
                
                pred_idx = pred_idx[keep]
            
            self.trackCount, new_log_vars, cls_loss, (trk_id_preds, score_preds, bbox_preds, cls_preds)  = \
                self.tracker.step(ego=self.ego,
                                    net=net,
                                    timestep=self.timestep,
                                    points=points,
                                    pred_cls=bbox_list[i][2][pred_idx],
                                    bev_feats=pts_feats[0][i,...],
                                    det_feats=merged[pred_idx],
                                    point_cloud_range=self.point_cloud_range,
                                    bbox=bbox_list[i][0][pred_idx] if len(pred_idx) > 0 else EmptyBBox(torch.empty([0,9],dtype=torch.float32,device=device)),
                                    det_confidence=bbox_list[i][1][pred_idx],
                                    trackCount=self.trackCount,
                                    device=device,
                                    last_in_scene=last_in_scene[i],
                                    sample_token=img_metas[i]['sample_idx'],
                                    gt_labels=gt_labels[i][gt_idx],
                                    gt_bboxes=gt_bboxes[i][gt_idx],
                                    gt_tracks=gt_tracks[i][gt_idx],
                                    gt_futures=gt_futures[i][gt_idx] if gt_futures[i].shape != torch.Size([]) else gt_futures,
                                    gt_pasts=gt_pasts[i][gt_idx] if gt_pasts[i].shape != torch.Size([]) else gt_pasts,
                                    gt_track_tte=gt_track_tte[i][gt_idx],
                                    output_preds=output_preds,
                                    return_loss=return_loss,)

            try:
                tracking_losses[i] = tracking_losses[i] + cls_loss
            except IndexError: 
                tracking_losses.append(cls_loss)

            update_log_vars(log_vars,new_log_vars)

            if output_preds:
                batch_track_preds.append(trk_id_preds.clone().detach())
                batch_track_score.append(score_preds.clone().detach())
                batch_track_bbox.append(bbox_preds.clone().detach())
                batch_track_labels.append(cls_preds.clone().detach())


            if output_preds:
                all_track_preds.append(catHandler(tensor_list=batch_track_preds,device=device)) 
                all_track_score.append(catHandler(tensor_list=batch_track_score,device=device))
                all_track_bbox.append(catHandler(tensor_list=batch_track_bbox,device=device))
                all_track_labels.append(catHandler(tensor_list=batch_track_labels,device=device))
            
        if last_in_scene[-1]:
            log_vars.update(self.get_scene_metrics(eval=False))

        if self.verbose:
            print("{} Leaving addDetection after {} seconds.".format(self.log_msg(),time.time() - t1))

        #TODO return the global tracking ids of objects in each frame
        return tracking_losses, log_vars, (all_track_preds, all_track_score, all_track_bbox, all_track_labels,)



    def addDetection_test(self,net,points,queries,pts_feats,bbox_list,img_metas,ego_update,
                          last_in_scene,first_in_scene,device,padded,
                          gt_labels=None,gt_bboxes=None,gt_tracks=None,gt_futures=None,gt_pasts=None,gt_track_tte=None,
                          return_loss=False,output_preds=True):
        """ Adds the detections from the current timestep.
        
        Args:
            pts_feats (torch.Tensor): BEVMap features
            bbox_list (torch.Tensor): label prediction for each 
                detection
            img_metas (list[dict]): Meta information of each sample.
                Defaults to None
            first_in_scene (list[bool]): Whether the corresponding 
                sample is the first in a scene
            first_in_scene (list[bool]): Whether the corresponding 
                sample is the last in a scene
            output_preds (bool): If True outputs tracking predictions o/w None
            return_loss (bool): If True computes the tracking loss o/w None

        Returns:
            tracking_losses (dict): losses computed from the association matrix
                for each class and each sample
        """

        batch,f,h,w = pts_feats[0].shape
        all_track_preds = [] 
        all_track_score = [] 
        all_track_bbox = [] 
        all_track_labels = []
        log_vars = {}

        for i in range(batch):
            if first_in_scene[i]:
                self.reset(net,device)

            self.timestep += 1
          
            self.ego.add_timestep(
                **{k:x[i]for k,x in ego_update.items()},
                device=device
            )

            if bbox_list[i][1].size(0) == 0:
                merged = torch.tensor([],device=device)
            else:
                merged, _, _ = net.getMergedFeats(
                    ego=self.ego,
                    pred_cls=bbox_list[i][2],
                    bbox_feats=bbox_list[i][0].tensor.to(device),
                    queries=queries[i],
                    bev_feats=pts_feats[0][i,...],
                    bbox_side_and_center=get_bbox_sides_and_center(bbox_list[i][0]),
                    confidence_scores=bbox_list[i][1].to(device),
                    point_cloud_range=self.point_cloud_range,
                    device=device
                )

            batch_track_preds = []
            batch_track_score = []
            batch_track_bbox = []
            batch_track_labels = []


            if padded[i]:
                all_track_preds.append(catHandler(tensor_list=batch_track_preds,device=device)) 
                all_track_score.append(catHandler(tensor_list=batch_track_score,device=device))
                all_track_bbox.append(catHandler(tensor_list=batch_track_bbox,device=device))
                all_track_labels.append(catHandler(tensor_list=batch_track_labels,device=device))
                continue



            pred_idx = torch.where(reduce(torch.logical_or,
                                        [bbox_list[i][2] == self.cls_to_idx[cls_] for cls_ in self.tracked_classes]))[0]
            gt_idx = torch.where(reduce(torch.logical_or,
                                        [gt_labels[i] == self.cls_to_idx[cls_] for cls_ in self.tracked_classes]))[0]

            if len(pred_idx) > 0 and self.use_det_nms:
                keep, suppress =  non_max_suppression(bboxes=bbox_list[i][0][pred_idx].tensor, 
                                                        confidence_scores=bbox_list[i][1][pred_idx], 
                                                        suppress_threshold=self.det_nms_threshold, 
                                                        ioucal=self.det_nms_ioucal,
                                                        device=device,
                                                        pred_labels=bbox_list[i][2][pred_idx])
                pred_idx = pred_idx[keep]


            if gt_labels is not None:
                gt_idx = torch.where(reduce(torch.logical_or,
                                        [gt_labels[i] == self.cls_to_idx[cls_] for cls_ in self.tracked_classes]))[0]

                gt_bboxes_=gt_bboxes[i][gt_idx]
                gt_tracks_=gt_tracks[i][gt_idx]
                gt_futures_=gt_futures[i][gt_idx] if gt_futures[i].shape != torch.Size([]) else gt_futures
                gt_pasts_=gt_pasts[i][gt_idx] if gt_pasts[i].shape != torch.Size([]) else gt_pasts
                gt_track_tte_=gt_track_tte[i][gt_idx]
                gt_labels_=gt_labels[i][gt_idx]
            else:
                gt_bboxes_=None
                gt_tracks_=None
                gt_futures_=None
                gt_pasts_=None
                gt_track_tte_=None
                gt_labels_=None

            self.trackCount, new_log_vars, losses, (trk_id_preds, score_preds, bbox_preds, cls_preds) = \
                self.tracker.step(ego=self.ego,
                                    net=net,
                                    timestep=self.timestep,
                                    points=points,
                                    pred_cls=bbox_list[i][2][pred_idx],
                                    bev_feats=pts_feats[0][i,...],
                                    det_feats=merged[pred_idx],
                                    point_cloud_range=self.point_cloud_range,
                                    bbox=bbox_list[i][0][pred_idx] if len(pred_idx) > 0 else EmptyBBox(torch.empty([0,9],dtype=torch.float32,device=device)),
                                    det_confidence=bbox_list[i][1][pred_idx],
                                    trackCount=self.trackCount,
                                    device=device,
                                    last_in_scene=last_in_scene[i],
                                    sample_token=img_metas[i]['sample_idx'],
                                    gt_labels=gt_labels_,
                                    gt_bboxes=gt_bboxes_,
                                    gt_tracks=gt_tracks_,
                                    gt_futures=gt_futures_,
                                    gt_pasts=gt_pasts_,
                                    gt_track_tte=gt_track_tte_,
                                    output_preds=output_preds,
                                    return_loss=return_loss)

            update_log_vars(log_vars,new_log_vars)

            batch_track_preds.append(trk_id_preds.clone().detach())
            batch_track_score.append(score_preds.clone().detach())
            batch_track_bbox.append(bbox_preds.clone().detach())
            batch_track_labels.append(cls_preds.clone().detach())


            all_track_preds.append(catHandler(tensor_list=batch_track_preds,device=device)) 
            all_track_score.append(catHandler(tensor_list=batch_track_score,device=device))
            all_track_bbox.append(catHandler(tensor_list=batch_track_bbox,device=device))
            all_track_labels.append(catHandler(tensor_list=batch_track_labels,device=device))

        if last_in_scene[-1]:
            log_vars.update(self.get_scene_metrics(eval=True))

              
        loss = None
        return loss, log_vars, all_track_preds, all_track_score, all_track_bbox, all_track_labels


    def get_scene_metrics(self,eval=False):
        metrics = {}
        # for cls in self.tracked_classes:
        #     metrics.update(self.trackers[cls].get_scene_metrics(eval=eval))
        metrics.update(self.tracker.get_scene_metrics(eval=eval))
        return metrics



    def update_grad_monitor(self,neptune=None):
        """Updates a list tracking the norm of
        current object parameter's gradients"""
        
        # print('update_grad_monitor',neptune)
        if self.gradMonitor == None:
            self.gradMonitor = {}
        paramMap = getParamInfo(self,'',grad=True)
        for k, v in paramMap.items():
            try:
                # print(k,v)
                if neptune:
                    neptune['grad/'+k].log(v.L1Norm)
                self.gradMonitor[k].append(v)
            except KeyError:
                self.gradMonitor[k] = [v]

    def update_param_monitor(self,neptune=None):
        """Updates a list tracking the norms of
        current object's parameters"""
        # print('inside_param_monitor',neptune)
        if self.paramMonitor == None:
            self.paramMonitor = {}
        paramMap = getParamInfo(self,'')
        for k, v in paramMap.items():
            try:
                # print(k,v)
                if neptune:
                    neptune['param/'+k].log(v.L1Norm)
                self.paramMonitor[k].append(v)
            except KeyError:
                self.paramMonitor[k] = [v]
        

def catHandler(tensor_list,device):
    if len(tensor_list) == 0:
        return torch.tensor([],dtype=torch.long,device=device)
    else:
        return torch.cat(tensor_list)