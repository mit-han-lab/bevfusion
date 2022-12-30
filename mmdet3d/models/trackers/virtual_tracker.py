import copy
import torch
# import pytorch3d

import torch.nn as nn
import numpy as np

from mmdet3d.models import TRACKERS
from mmdet3d.models  import builder
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS, build_iou_calculator
from mmdet3d.core.bbox.iou_calculators import BboxOverlapsNearest3D
from mmdet3d.ops import points_in_boxes_batch
from mmdet3d.core import LiDARInstance3DBoxes

from .track import Track
from .tracking_helpers import get_bbox_sides_and_center, EmptyBBox, get_cost_mat_viz, log_mistakes
from .transforms import affine_transform, apply_transform

from telnetlib import PRAGMA_HEARTBEAT
from functools import reduce
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
# from pytorch3d.structures.pointclouds import Pointclouds


def get_affine_torch(rotation,translation,device):
    """
    axis_angle (torch.tensor (B,1)): Rotations given as a vector in axis angle form, as a tensor of shape (…, 3), 
        where the magnitude is the angle turned anticlockwise in radians around the vector’s direction.
    """
    b,_ = rotation.shape
    out = torch.zeros((b,4,4,), dtype=torch.float32, device=device)
    out[:,3,3] = 1.0
    rot = pytorch3d.transforms.axis_angle_to_matrix(rotation)
    out[:,:3,:3] = rot
    out[:,[0,1,2],3] = translation
    return out


    
def get_pts_in_bbox_batched(points,bbox,num_dets,device):
    pib_idx = points_in_boxes_batch(points[0][: ,:3].unsqueeze(0), temp.tensor.unsqueeze(0),)
    pib = [points[0][pib_idx[0,:,i].bool() ,:3] for i in range(num_dets)]
    lengths = torch.tensor([x.size(0) for x in pib],device=device)
    rot = torch.zeros((num_dets,3),dtype=torch.float32,device=device)
    rot[:,2] = (temp.tensor[:,6] + np.pi/2 ) * -1.
    aff = get_affine_torch(rotation=rot,
                            translation=temp.tensor[:,:3],
                            device=device)
    pc = Pointclouds(pib)
    padded = pc.points_padded()
    padded = torch.cat([padded, torch.ones((padded.size(0),padded.size(1),1),device=device)],dim=2)
    trnsf = torch.bmm(torch.linalg.inv(aff),padded.reshape(num_dets,4,-1))
    return [trnsf[i,:x,:] for i,x in enumerate(lengths)]

def get_pts_in_bbox_seq(points,bbox,num_dets,device):
    pib_idx = points_in_boxes_batch(points[0][: ,:3].unsqueeze(0), bbox.tensor.unsqueeze(0),)
    # print('pib_idx.shape',pib_idx.shape)
    # print(pib_idx.bool())
    pib = [points[0][pib_idx[0,:,i].bool() ,:3] for i in range(num_dets)]
    pts = []
    for i in range(num_dets):
        temp_bbox = bbox.tensor[i,:].cpu().numpy()
        aff_r = affine_transform(
                rotation=np.roll(Quaternion(axis=[0, 0, 1], angle=((temp_bbox[6] + np.pi/2 ) * -1)).elements,-1),
                rotation_format='quat',
                translation=temp_bbox[:3],
            )
        pts.append(apply_transform(np.linalg.inv(aff_r),pib[i].cpu().numpy()))
    return pts



@IOU_CALCULATORS.register_module()
class Center2DRange(object):
    """Within 2D range"""

    def __init__(self, distance=2, coordinate='lidar'):
        # assert coordinate in ['camera', 'lidar', 'depth']
        self.distance = distance

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """assumes xy are the first two coordinates
        """
        
        iou = torch.cdist(bboxes1[:,:2],bboxes2[:,:2],p=2) 
        # idx1,idx2 = torch.where(iou < self.distance)
        


        return iou #, idx1, idx2

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str


@TRACKERS.register_module()
class VirtualTracker(nn.Module):
    ioucal = BboxOverlapsNearest3D('lidar')


    def log_str(self,timestep=None):
        if timestep == None:
            return "[VirtualTracker | Rank:{}]".format(self.rank)
        else:
            return "[VirtualTracker | Rank:{}, timestep: {}]".format(self.rank,timestep)

    def __init__(self, 
                 cls,
                 use_pc_feats=False,
                 frameLimit=10,
                 tp_threshold=0.01,
                 rank=-1, 
                 track_supervisor=None, 
                 gt_testing=False,
                 box_output_mode='det',
                 use_nms=True,
                 suppress_threshold=0.15,
                 associator=None,
                 incrementor=None, 
                 modifier=None, 
                 updater=None,
                 verbose=False,
                 use_mp_feats=True,
                 detection_decisions=['det_false_positive','det_newborn'], 
                 tracking_decisions=[],
                 teacher_forcing=False,
                 ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                 visualize_cost_mat=False,
                 propagation_method='velocity' # 'velocity' or 'future'
                 ):
        """The Track Manager implements the logic for tracking objects of the same class

        args:
            cls (string): class name
            frameLimit (int): The number of frames to persist unmatched tracks 
            tp_threshold (float): tracking score threshold for true positives
            rank (int): rank of the process
        
        """
        
        super().__init__()
        # super(TrackManager, self).__init__()
        self.cls = cls
        self.tp_threshold = tp_threshold
        self.verbose=verbose
        self.rank=rank
        self.frameLimit = frameLimit
        self.suppress_threshold = suppress_threshold
        self.use_nms = use_nms
        self.teacher_forcing = teacher_forcing
        self.visualize_cost_mat = visualize_cost_mat
        self.ioucal = build_iou_calculator(ioucal)
        self.box_output_mode = box_output_mode

        self.gt_testing = gt_testing
        self.use_mp_feats = use_mp_feats #update detection features with their representation after transformer
        self.use_pc_feats = use_pc_feats #use point features from cropped PC

        self.propagation_method = propagation_method

        self.tracks = []
        self.activeTracks = [] # tensor index : track index
        self.unmatchedTracks = {} # tensor index : track index
        self.decomTracks = [] # tensor index : track index

        self.mistakes_track, self.mistakes_det = {},{}

        # self.save_feats = []

        self.trkid_to_gt_trkid = None
        self.trkid_to_gt_tte = None
        self.epoch = 0 

        self.detection_decisions = sorted(detection_decisions)
        self.tracking_decisions = sorted(tracking_decisions) #need to sort for canonical ordering
        self.dd_num = len(detection_decisions)
        self.td_num = len(tracking_decisions)

        self.logging = {}
        for k in ['total','det_match'] + self.detection_decisions + self.tracking_decisions:
            self.logging[k+'_correct'] = 0
            self.logging[k+'_gt'] = 0.00000000001 # avoid divide by zero
            if k != 'total':
                self.logging[k+'_num_pred'] = 0.00000000001 # avoid divide by zero

        
        self.tsup = builder.build_supervisor(track_supervisor)
        self.associator = builder.build_tracker(associator)
        self.incrementor = builder.build_tracker(incrementor)
        self.modifier = builder.build_tracker(modifier)
        self.updater = builder.build_tracker(updater)


    def set_epoch(self,epoch,max_epoch):
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.modifier.set_epoch(epoch,max_epoch)

    def reset(self,net,device):
        """Resets the TrackManager to default for the next sequence"""
        del self.tracks
        self.tracks = []
        del  self.activeTracks
        self.activeTracks = []
        del  self.unmatchedTracks
        self.unmatchedTracks = {}
        del  self.decomTracks
        self.decomTracks = []
        del self.trkid_to_gt_trkid
        self.trkid_to_gt_trkid = torch.empty((0,),dtype=torch.long,device=device)
        del self.trkid_to_gt_tte
        self.trkid_to_gt_tte = torch.empty((0,),dtype=torch.long,device=device)
        self.incrementor.reset(net,device)
        del self.logging
        self.logging = {}
        for k in ['total','det_match'] + self.detection_decisions + self.tracking_decisions:
            self.logging[k+'_correct'] = 0
            self.logging[k+'_gt'] = 0.00000000001 # avoid divide by zero
            self.logging[k+'_num_pred'] = 0.00000000001 # avoid divide by zero

    def gatherAny(self,feats,index):
        """Indexes into the appropriate tensor to gather the desired
        freatures."""
        return self.incrementor.gatherAny(feats,index)

    def gatherActive(self,feats):
        """Indexes into the appropriate tensor to gather the desired
        freatures."""
        return self.incrementor.gatherAny(feats,self.activeTracks)


    def get_iou_idx(self,bbox,gt_bboxes,device,cls1=None,cls2=None,):
        if cls1 != None and cls2 != None and len(cls1) > 0 and len(cls2) > 0:
            # print(cls1.shape)
            # print(cls2.shape)
            # print(bbox.tensor.shape)
            # print(gt_bboxes.tensor.shape)
            assert len(cls1) == bbox.tensor.size(0)
            assert len(cls2) == gt_bboxes.tensor.size(0)
            mask = torch.zeros((bbox.tensor.size(0),gt_bboxes.tensor.size(0),),dtype=torch.float32,device=device)
            cp = torch.cartesian_prod(torch.arange(0,bbox.tensor.size(0)), torch.arange(0,gt_bboxes.tensor.size(0)))
            mask[cp[:,0],cp[:,1]] = (cls1.long()[cp[:,0]] != cls2.long()[cp[:,1]]).float() * 10000.0
        else:
            mask = torch.zeros((bbox.tensor.size(0),gt_bboxes.tensor.size(0),),dtype=torch.float32,device=device)
            

        if type(self.ioucal) == Center2DRange:

            det_iou = self.ioucal(bbox.tensor,gt_bboxes.tensor)
            det_iou = det_iou + mask

            tp_det_idx, tp_gt_idx = linear_sum_assignment(det_iou.cpu().numpy())
            tp_det_idx = torch.from_numpy(tp_det_idx).to(device)
            tp_gt_idx = torch.from_numpy(tp_gt_idx).to(device)
        
            matches = torch.where(det_iou[tp_det_idx,tp_gt_idx] < self.ioucal.distance)
            tp_det_idx = tp_det_idx[matches]
            tp_gt_idx = tp_gt_idx[matches]

        else:
            det_iou = self.ioucal(bbox.tensor,gt_bboxes.tensor) * -1 
            det_iou = det_iou + mask
            # det_iou[torch.where(det_iou > ( self.tp_threshold * -1) )] = 10000.
            tp_det_idx, tp_gt_idx = linear_sum_assignment(det_iou.cpu().numpy())
            tp_det_idx = torch.from_numpy(tp_det_idx).to(device)
            tp_gt_idx = torch.from_numpy(tp_gt_idx).to(device)

            matches = torch.where(det_iou[tp_det_idx,tp_gt_idx] < ( self.tp_threshold * -1))
            tp_det_idx = tp_det_idx[matches]
            tp_gt_idx = tp_gt_idx[matches]

        


        return det_iou[tp_det_idx,tp_gt_idx], tp_det_idx, tp_gt_idx


    def non_max_suppression(self,device):
        """nms on the active tracks"""
        if not self.use_nms:
            return

        if len(self.tracks) == 0 or len(self.activeTracks) == 0:
            return

        classes = torch.tensor([self.tracks[i].cls[-1] for i in self.activeTracks],dtype=torch.long, device=device)
        mask = torch.zeros((len(self.activeTracks),len(self.activeTracks)),dtype=torch.float32, device=device)
        cp = torch.cartesian_prod(torch.arange(0,len(self.activeTracks)), torch.arange(0,len(self.activeTracks))).to(device)
        mask[cp[:,0],cp[:,1]] = (classes[cp[:,0]] != classes[cp[:,1]]).float() * -10000.0

        bboxes = torch.stack([self.tracks[i].det_bboxes[-1] for i in self.activeTracks], dim=0)
        iou = VirtualTracker.ioucal(bboxes,bboxes)
        iou = iou + mask



        iou = torch.triu(iou, diagonal=1) # suppress duplicate entires 
        idx1,idx2 = torch.where(iou > self.suppress_threshold)
        # print('to_suppress: ',idx1,idx2)

        track_score = torch.stack([len(self.tracks[i].det_bboxes) + self.tracks[i].scores[-1] for i in self.activeTracks], dim=0)
        scores = track_score[idx1] - track_score[idx2]
        suppress = [self.activeTracks[x] for x in idx1[torch.where(scores <= 0)]] + [self.activeTracks[x] for x in idx2[torch.where(scores > 0)]]
        # print("[actually suppressing]",suppress)

        self.activeTracks = [x for x in self.activeTracks if x not in suppress]
        self.decomTracks += suppress


    def get_dets_to_trk_idx(self,num_dets,active_tracks,decisions,device):
        """Returns a tensor mapping each incoming detection to its global track ID. All 
        detections are kept, but not all are made into active tracks."""
        dets_to_trk_idx = torch.zeros(num_dets,dtype=torch.int64,device=device)
        dets_to_trk_idx[decisions['det_match']] = torch.tensor(active_tracks,dtype=torch.long,device=device)[decisions['track_match']]
        offset = 0
        for k in self.detection_decisions:
            
            dets_to_trk_idx[decisions[k]] = torch.tensor([len(self.tracks) + offset + ii  for ii in range(len(decisions[k]))], #assign new track indices to new dets
                                                            dtype=torch.long,
                                                            device=device)
            offset += len(decisions[k])

        return dets_to_trk_idx

    def get_active_bboxes(self,timestep):
        try:
            return [self.tracks[i][timestep] for i in self.activeTracks]
        except KeyError:
            print("Rank:",self.rank)
            print("timestep:",timestep)
            print("active_tracks",self.activeTracks)
            exit(0)


    def get_track_det_distances(self,bbox,ego,timestep,device):
        if bbox.tensor.size(0) == 0 or len(self.activeTracks) == 0:
            return []

        dets_xyz = bbox.tensor[:,:3]
        dets_prev = ego.transform_over_time(dets_xyz.cpu().numpy(),from_=timestep,to_=timestep-1)
        track_xyz = torch.stack(self.get_active_bboxes(timestep=timestep-1))[:,:2]
        dists = torch.cdist(track_xyz,torch.from_numpy(dets_prev.astype(np.float32)).to(device)[:,:2],p=2.0)
        return dists
        
    def update_gt_track_mapping(self,gt_track_tte,gt_tracks,tp_det_idx,tp_gt_idx,dets_to_trk_idx,device):
        """Updates the mapping from track ID to GT track ID and GT track TTE

        Takes indices of the kept detections at the current timestep
        and stores them at index positions correspoiinding to their track 
        id in tensors for fast lookup later.

        Args:
            gt_track_tte (torch.Tensor): tensor of GT track TTEs.
            gt_tracks (torch.Tensor): tensor of GT track IDs.
            tp_det_idx (torch.Tensor): indices of TP detections corresponding to tp_gt_idx.
            tp_gt_idx (torch.Tensor): indices of GT tracks corresponding to TP detections.
            dets_to_trk_idx (torch.Tensor): tensor mapping each detection to its global track ID.
            device (torch.device): device to store tensors on.
        
        """
        #-2: newborn -1: false positive
        update_trkid = torch.full((self.incrementor.track_feats.size(0),),-2,device=device)
        update_tte = torch.full((self.incrementor.track_feats.size(0),),-1,device=device)

        update_trkid[:self.trkid_to_gt_trkid.size(0)] = self.trkid_to_gt_trkid
        update_tte[:self.trkid_to_gt_tte.size(0)] += self.trkid_to_gt_tte #decrement one timestep

        #get False Positive detections
        temp = torch.full_like(dets_to_trk_idx,1)
        temp[tp_det_idx] = 0
        fp_det_idx = torch.where(temp == 1)[0]
        
        try:
            assert ( len(fp_det_idx) + len(tp_det_idx) ) == len(dets_to_trk_idx)
        except AssertionError:
            print("fp_det_idx:",len(fp_det_idx),fp_det_idx)
            print("tp_det_idx:",len(tp_det_idx),tp_det_idx)
            print("dets_to_trk_idx:",len(dets_to_trk_idx),dets_to_trk_idx)
            print("Temp",temp.shape)
            print("len(temp) - len(tp_det_idx) - len(fp_det_idx[0])",len(temp) - len(tp_det_idx) - len(fp_det_idx[0]))
            exit(0)


        # tp1_to_tp2 = self.log_mistakes(prev_tp_det_trkid=update_trkid[dets_to_trk_idx[tp_det_idx]],
        #                                prev_tp_det_tte=update_tte[dets_to_trk_idx[tp_det_idx]] ,
        #                                prev_fp_det_trkid=update_trkid[dets_to_trk_idx[fp_det_idx]],
        #                                prev_fp_det_tte=update_tte[dets_to_trk_idx[fp_det_idx]],
        #                                update_tp_det_trkid=gt_tracks[tp_gt_idx],
        #                                update_tp_det_tte=gt_track_tte[tp_gt_idx],
        #                                tp_det_idx=tp_det_idx,
        #                                fp_det_idx=fp_det_idx,
        #                                dets_to_trk_idx=dets_to_trk_idx,
        #                                curr_dets_to_track_idx=None,
        #                                cost_mat=None,)


        #Update GT IDs of TP matches and newborns
        update_trkid[dets_to_trk_idx[tp_det_idx]] = gt_tracks[tp_gt_idx]
        update_tte[dets_to_trk_idx[tp_det_idx]] = gt_track_tte[tp_gt_idx]

        #Update GT IDs of FP matches
        update_trkid[dets_to_trk_idx[fp_det_idx]] = -1
        update_tte[dets_to_trk_idx[fp_det_idx]] = -1

        #set -2 to -1
        update_trkid[update_trkid == -2] = -1
 
        self.trkid_to_gt_trkid = update_trkid
        self.trkid_to_gt_tte = update_tte


    def step(self,ego,net,timestep,points,pred_cls,bev_feats,det_feats,point_cloud_range,bbox,trackCount,device,
             last_in_scene,det_confidence,sample_token=None,gt_labels=None,
             gt_bboxes=[],gt_tracks=None,output_preds=False,return_loss=True,gt_futures=None,
             gt_pasts=None,gt_track_tte=None):
        """Take one step forward in time by processing the current frame.

        Args:
            det_feats (torch.Tensor, required): Feature representation 
                for the detections computed by MLPMerge
            bbox (torch.Tensor, required): bounding boxes for each 
                detection
            lstm (nn.LSTM, required): lstm for processing tracks
            trackCount (int, required): the number of tracks currently
                allocated for this scene
            c0 (torch.Tensor): lstm initial cell state
            h0 (torch.Tensor): lstm initial hidden state
            sample_token (str, optional): token of the current detection
                used for computing performance metrics
            gt_bboxes (LiDARInstance3DBoxes): GT bounding boxes
            gt_tracks (torch.Tensor): GT tracks ids
            output_preds (bool): If True outputs tracking predictions o/w None
            return_loss (bool): If True compute the loss
        
        Returns: 
            trackCount (int): the new track count
            log_vars (dict): information to be logged 
            losses (dict): losses computed from the association matrix
            (torch.tensor) list of global tracking ids corresponding to the current 
                detections if output_preds is True, None otherwise

        """
        if self.verbose:
            print("{} entering step().".format(self.log_str(timestep)))


        assert timestep >= 0
        log_vars = {}
        num_tracks = len(self.activeTracks)
        num_dets = det_feats.size(0)


        # print("Timestep:{}, num_tracks:{}, num_dets:{}, rank:{}".format(timestep,num_tracks,num_dets,self.rank))

        self.log_update({'mean_num_tracks_per_timestep':torch.tensor([num_tracks],dtype=torch.float32),
                         'mean_num_dets_per_timestep':torch.tensor([num_dets],dtype=torch.float32)})
        
        trk_feats = self.gatherActive(['f'])
        trk_feats = trk_feats.clone()
        decision_feats = torch.tensor([num_tracks,num_dets],dtype=torch.float32,device=device)


        if self.use_pc_feats:
            import time
            temp = copy.deepcopy(bbox)
            temp.tensor = temp.tensor[:,:-2] # must be tensor of size 7
            t1 = time.time()
            import timeit

            t1 = timeit.timeit(lambda : get_pts_in_bbox_batched(points,bbox=temp,num_dets=num_dets,device=device),number=1000)
            t2 = timeit.timeit(lambda : get_pts_in_bbox_seq(points,bbox=temp,num_dets=num_dets,device=device),number=1000)

            print(t1,t2,num_dets)
            print('div',t1/1000,t2/1000,num_dets)
            t2 = time.time()
            exit(0)



        supervise, trk_feats_mp, det_feats_mp = net.forward_association(trk_feats=trk_feats,
                                                                        det_feats=det_feats,
                                                                        decision_feats=decision_feats,
                                                                        tracking_decisions=self.tracking_decisions,
                                                                        detection_decisions=self.detection_decisions,
                                                                        device=device,)


        if self.use_mp_feats:
            #increment using message passing features
            trk_feats = trk_feats_mp
            det_feats = det_feats_mp


        if self.associator.use_distance_prior:
            track_det_dists = self.get_track_det_distances(bbox=bbox,
                                                           ego=ego,
                                                           timestep=timestep,
                                                           device=device) 
        else:
            track_det_dists = []

        
        class_mask = torch.zeros((num_tracks,num_dets),dtype=torch.float32,device=device)
        track_classes = torch.tensor([self.tracks[x].cls[-1] for x in self.activeTracks],dtype=torch.long,device=device)
        cp = torch.cartesian_prod(torch.arange(0,num_tracks), torch.arange(0,num_dets))
        class_mask[cp[:,0],cp[:,1]] = (track_classes[cp[:,0]] != pred_cls[cp[:,1]]).float() * 10000.0

        decisions, summary, (numpy_cost_mat, dd_mat, td_mat) = self.associator(supervise,
                                                                                tracker=self, 
                                                                                num_trk=num_tracks, 
                                                                                num_det=num_dets, 
                                                                                class_name=self.cls, 
                                                                                track_det_dists=track_det_dists, 
                                                                                class_mask=class_mask,
                                                                                device=device)
        log_vars.update(summary)


        if gt_tracks != None:
            tp_decisions, mod_decisions, tp_det_idx, tp_gt_idx  = self.modifier(tracker=self,
                                                                                bbox=bbox,
                                                                                gt_bboxes=gt_bboxes,
                                                                                gt_tracks=gt_tracks,
                                                                                pred_cls=pred_cls,
                                                                                gt_labels=gt_labels,
                                                                                prev_AT=self.activeTracks,
                                                                                decisions=decisions,
                                                                                device=device)

            self.log_update({f"num_TP_{k[4:]}":torch.tensor([len(v)],dtype=torch.float32) for k,v in tp_decisions.items() if k.startswith('pos')})
            self.log_update({f"num_dec_{k}":torch.tensor([len(v)],dtype=torch.float32) for k,v in decisions.items()})

            try:
                assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:9] == 'pos_track']) == \
                       sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']), 'track decisions issue'

                assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) == \
                       sum([x.size(0) for k,x in decisions.items() if k[:3] == 'det']), "detection decisions issue"
            except AssertionError as e:
                print(e)
                print("[AssertionError]  assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) ==sum([x.size(0) for k,x in decisions.items() if k[:3] == 'det'])")
                print('or')
                print("[AssertionError]  assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:9] == 'pos_track']) == sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track'])")

                print('In virtual tracker step()')
                print('      decisions:',[[k,x.size(0)] for k,x in decisions.items()],sum([x.size(0) for k,x in decisions.items()]))
                print('decisions det  : ',sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) )
                print('decisions track: ',sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']) )
                print('      tp_decisions:',[[k,x.size(0)] for k,x in tp_decisions.items() if k[:4] == 'pos_'],sum([x.size(0) for k,x in tp_decisions.items() if k[:4] == 'pos_']))
                print('tp_decisions det  : ',sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) )
                print('tp_decisions track: ',sum([x.size(0) for k,x in tp_decisions.items() if k[:9] == 'pos_track']) )

                for k,x in tp_decisions.items():
                    if k[:9] == 'pos_track':
                        print(k,x)
                        idx = torch.tensor(self.activeTracks)[x]
                        print('trkid_to_gt_trkid',self.trkid_to_gt_trkid[idx])
                        print('trkid_to_gt_tte',self.trkid_to_gt_tte[idx])

                print('gt_trk_id',gt_tracks)
                print('gt_trk_tte',gt_track_tte)

                print('timestep:',timestep)

                print('num_det',num_dets)
                print('num_track',num_tracks)
                exit(0)


        # if return_loss == False:
        #     temp = torch.full_like(num_dets,1)
        #     temp[tp_det_idx] = 0
        #     fp_det_idx = torch.where(temp == 1)[0]
        #     self.mistakes_track, self.mistakes_det = log_mistakes(tracker=self,
        #                                                           tp_decisions=tp_decisions,
        #                                                           decisions=decisions,
        #                                                           td_mat=td_mat,
        #                                                           dd_mat=dd_mat,
        #                                                           cost_mat=numpy_cost_mat,
        #                                                           num_track=num_tracks,
        #                                                           num_det=num_dets,
        #                                                           mistakes_track=self.mistakes_track,
        #                                                           mistakes_det=self.mistakes_det,
        #                                                           det_tp_idx=tp_det_idx,
        #                                                           active_tracks=self.activeTracks,
        #                                                           device=device,)
            

        if ( return_loss == True and self.teacher_forcing) or self.gt_testing:
            #teacher forcing
            decisions = mod_decisions


        if self.visualize_cost_mat:
            get_cost_mat_viz(tracker=self,
                             tp_decisions=tp_decisions,
                             decisions=decisions,
                             td_mat=td_mat,
                             dd_mat=dd_mat,
                             cost_mat=numpy_cost_mat,
                             num_track=num_tracks,
                             num_det=num_dets,
                             device=device,
                             save_dir='/mnt/bnas/tracking_results/cost_mats/{}.pdf'.format(sample_token),
                             show=False)



        
        if 'track_false_negative' in self.tracking_decisions:

            if len(decisions['track_false_negative'] ) > 0:
                # print(gt_bboxes.__dict__)

                bbox_list_fn = [torch.tensor(self.tracks[self.activeTracks[t]].transform_over_time(from_=timestep-1,to_=timestep,ego=ego,propagation_method=self.propagation_method)[1],
                                          dtype=torch.float32,
                                          device=device) # [0] for refined [1] for det
                                for t in decisions['track_false_negative'] ]
                bbox_list_fn = torch.stack(bbox_list_fn)
                bbox_list_fn = LiDARInstance3DBoxes(bbox_list_fn, box_dim=9, with_yaw=True, origin=(0.5, 0.5, 0))

                confidence_scores_fn = torch.stack([ self.tracks[self.activeTracks[t]].det_confidence[-1] for t in decisions['track_false_negative'] ])
                pred_cls_fn = torch.stack([self.tracks[self.activeTracks[t]].cls[-1] for t in decisions['track_false_negative'] ])

                false_negative_emb, _, _ = net.getMergedFeats(
                        ego=ego,
                        pred_cls=pred_cls_fn,
                        bbox_feats=bbox_list_fn.tensor.to(device),
                        bbox_side_and_center=get_bbox_sides_and_center(bbox_list_fn),
                        bev_feats=bev_feats,
                        confidence_scores=confidence_scores_fn,
                        point_cloud_range=point_cloud_range,
                        device=device,
                    )


                if self.use_mp_feats:
                    false_negative_emb = false_negative_emb + trk_feats[decisions['track_false_negative'],:]


                    
                # false_negative_emb = net.false_negative_emb.weight.repeat(decisions['track_false_negative'].size(0),1)
                false_negative_idx = det_feats.size(0) + torch.arange(0,decisions['track_false_negative'].size(0),device=device)
            else:
                false_negative_idx = torch.empty((0),dtype=torch.long,device=device)
                false_negative_emb = torch.empty((0,det_feats.size(1)),dtype=torch.float32,device=device)
            #take an LSTM step based on the new tracks
            self.incrementor(net=net,
                            active_tracks=self.activeTracks,
                            hidden_idx=torch.cat([decisions['track_match'],decisions['track_false_negative']]),
                            increment_hidden_idx=torch.cat([decisions['det_match'],false_negative_idx]),
                            new_idx=torch.cat([decisions[k] for k in self.detection_decisions]),
                            feats=torch.cat([det_feats,false_negative_emb],dim=0),
                            device=device)
        else:
            self.incrementor(net=net,
                            active_tracks=self.activeTracks,
                            hidden_idx=decisions['track_match'],
                            increment_hidden_idx=decisions['det_match'],
                            new_idx=torch.cat([decisions[k] for k in self.detection_decisions]),
                            feats=det_feats,
                            device=device)


        #TODO find a better way to retain active tracks
        prev_AT = copy.deepcopy(self.activeTracks)
        dets_to_trk_idx = self.get_dets_to_trk_idx(det_feats.size(0), prev_AT, decisions, device)

        if gt_tracks != None:#train vs testing
            self.update_gt_track_mapping(gt_track_tte,gt_tracks,tp_det_idx,tp_gt_idx,dets_to_trk_idx,device)

                                                
        assert self.incrementor.track_feats.size(0) >= dets_to_trk_idx.size(0)

        # print(self.incrementor.track_feats.shape)
        current_trk_feats = self.gatherAny('f',dets_to_trk_idx)
        forecast_preds = net.MLPPredict(current_trk_feats)
        refine_preds = net.MLPRefine(current_trk_feats)
        refine_preds = torch.cat([refine_preds[:,:-1],torch.sigmoid(refine_preds[:,-1]).unsqueeze(1)],dim=1)

        #update the actual Track objects representing the tracks
        trackCount, tracks_to_output = self.updater(tracker=self,
                                                    pred_cls=pred_cls,
                                                    det_confidence=det_confidence,
                                                    trackCount=trackCount,
                                                    bbox=bbox,
                                                    decisions=decisions,
                                                    sample_token=sample_token,
                                                    refine_preds=refine_preds,
                                                    forecast_preds=forecast_preds,
                                                    timestep=timestep,
                                                    ego=ego,
                                                    device=device)




        assert len(self.tracks) >= dets_to_trk_idx.size(0)
        assert dets_to_trk_idx.size(0) == refine_preds.size(0) 

        losses = torch.tensor(0.,device=device, requires_grad=True)
        if return_loss: #only calculate loss info during training
            # print("In supervise tracking seciton")

            assert len(prev_AT) == trk_feats.size(0)
            assert dets_to_trk_idx.size(0) == det_feats.size(0)
            assert dets_to_trk_idx.size(0) == bbox.tensor.size(0)

            
            association_loss, summary, log = self.tsup.supervise_association(
                    tracker=self,
                    tp_decisions=tp_decisions,
                    supervise=supervise,
                    device=device,
                    return_loss=return_loss
            )
            log_vars.update(summary)
            self.log_update(log)


            if len(dets_to_trk_idx) == 0:
                refined_bboxes = torch.empty([0,9],dtype=torch.float32,device=device)
            else:
                refined_bboxes = torch.cat([self.tracks[i].refined_bboxes[-1].unsqueeze(0) for i in dets_to_trk_idx])

            refine_loss, summary, log = self.tsup.supervise_refinement(
                                            tp_det_idx=tp_det_idx,
                                            tp_gt_idx=tp_gt_idx,
                                            refine_preds=refine_preds,
                                            refined_bboxes=refined_bboxes,
                                            bbox=bbox,
                                            gt_pasts=gt_pasts,
                                            gt_bboxes=gt_bboxes,
                                            device=device,
                                            return_loss=return_loss)
            log_vars.update(summary)
            self.log_update(log)

            forecast_loss, summary, log = self.tsup.supervise_forecast(
                                        tp_det_idx=tp_det_idx,
                                        tp_gt_idx=tp_gt_idx,
                                        forecast_preds=forecast_preds,
                                        gt_futures=gt_futures,
                                        device=device,
                                        log_prefix='track_',
                                        return_loss=return_loss)
            log_vars.update(summary)
            self.log_update(log)
            
            if gt_bboxes.tensor.nelement() == 0:
                #Naive workaround to avoid unused paramters error in torch
                refine_loss = torch.tensor(0.,device=device, requires_grad=True)# torch.tensor net.MLPRefine(torch.randn((10,net.MLPRefine[0].in_features),device=device)).sum() * 0.0
                forecast_loss = torch.tensor(0.,device=device, requires_grad=True)# net.MLPPredict(torch.randn((10,net.MLPPredict[0].in_features),device=device)).sum() * 0.0
                association_loss = association_loss * 0.0

            losses = refine_loss + forecast_loss + association_loss


        
        trk_id_preds, score_preds, bbox_preds, cls_preds = self.get_global_track_id_preds(tracks_to_output, device=device, output_preds=output_preds)

        
        if self.verbose:
            print("{} ending step().".format(self.log_str(timestep)))

        return trackCount, log_vars, losses, (trk_id_preds, score_preds, bbox_preds, cls_preds,)

    def log_update(self,x):
        for k in x: 
            try:
                self.logging[k].append(x[k])
            except KeyError:
                self.logging[k] = [x[k]]
        

    def get_scene_metrics(self,eval=False):
        metrics = {}
        if eval:
            metrics.update({'eval_'+ k:v for k,v in self.logging.items()})
            len_tracks = [len(trk.refined_bboxes) for trk in self.tracks]
            metrics[f'eval_len_tracks'] = len_tracks
            greater_than_one_tracks = [x for x in len_tracks if x > 1]
            metrics[f'eval_greater_than_one_tracks'] = greater_than_one_tracks
            metrics[f'eval_scene_total_tracks'] = len(self.tracks)
            metrics.update({'mistakes_track_'+ k:v for k,v in self.mistakes_track.items()})
            metrics.update({'mistakes_det_'+ k:v for k,v in self.mistakes_det.items()})

            return metrics

        else:
            len_tracks = [len(trk.refined_bboxes) for trk in self.tracks]
            metrics[f'mean_track_length_{self.cls}'] = np.mean(len_tracks)
            metrics[f'median_track_length_{self.cls}'] = np.median(len_tracks)

            greater_than_one_tracks = [x for x in len_tracks if x > 1]
            if greater_than_one_tracks != []:
                metrics[f'mean_track_length_>1_{self.cls}'] = np.mean(greater_than_one_tracks)

            metrics[f'total_tracks_{self.cls}'] = len(self.tracks)

            for k in ['total','det_match'] + self.detection_decisions + self.tracking_decisions:
                if k == 'total':
                    metrics[f'acc_{k}_{self.cls}'] = self.logging[k+'_correct'] / (self.logging[k+'_gt'] + 0.000000000001)
                else:
                    metrics[f'recall_{k}_{self.cls}'] = self.logging[k+'_correct'] / ( self.logging[k+'_gt'] + 0.000000000001)
                    metrics[f'precision_{k}_{self.cls}'] = self.logging[k+'_correct'] / ( self.logging[k+'_num_pred'] + 0.000000000001)

                    add = metrics[f'recall_{k}_{self.cls}'] + metrics[f'precision_{k}_{self.cls}'] + 0.000000000001
                    mul = metrics[f'recall_{k}_{self.cls}'] * metrics[f'precision_{k}_{self.cls}']
                    metrics[f'f1_{k}_{self.cls}'] = 2 * (mul/add)
                    

            total_tp_decisions = torch.cat([torch.cat(x) for k,x in self.logging.items() if k.startswith('num_TP')]).sum()
            

            for k in self.logging:

                if k.startswith('num_TP') or k.startswith('num_dec'):
                    metrics[f'%_{k[4:]}'] = torch.cat(self.logging[k]).sum() / (total_tp_decisions + 0.000000000001)

                elif not ( k.endswith('_gt') or k.endswith('_correct') or k.endswith('_num_pred')):
                    # print(self.logging[k])
                    # print([x.shape for x in self.logging[k]])
                    metrics[k] = torch.cat(self.logging[k]).mean().item()
            

        return metrics

    

    def get_global_track_id_preds(self, tracks_to_output, device, output_preds=True):
        """
        Args:
            tracks_to_output (torch.Tensor): list of class-specific track
                ids corresponding to output at the current frame.
            output_preds (bool): whether to compute predictions or not.
        
        Returns:
            (torch.tensor) list of global tracking ids corresponding to the current 
                detections if output_preds is True, None otherwise.
        """
        if output_preds:
            trk_id_preds = torch.tensor([self.tracks[i].id for i in tracks_to_output],device=device)
            score_preds = torch.tensor([self.tracks[i].det_confidence[-1] for i in tracks_to_output],device=device)
            cls_preds = torch.tensor([self.tracks[i].cls[-1] for i in tracks_to_output],device=device)


            if len(tracks_to_output) == 0:
                bbox_preds = torch.tensor([],device=device)
            else:
                if self.box_output_mode.lower() == 'det':
                    bbox_preds = torch.stack([self.tracks[i].det_bboxes[-1] for i in tracks_to_output],dim=0)
                elif  self.box_output_mode.lower() == 'track':
                    bbox_preds = torch.stack([self.tracks[i].refined_bboxes[-1] for i in tracks_to_output],dim=0)
                else:
                    raise ValueError("Incorrect box_output_mode")

            return trk_id_preds, score_preds, bbox_preds, cls_preds
        else:
            return None, None, None, None


































    #deprecated
    def initTracks(self,ego,net,timestep,det_feats,bbox,trackCount,device,det_confidence,
                    sample_token=None, gt_bboxes=[],gt_tracks=None,output_preds=False,
                    return_loss=True,gt_futures=None,gt_pasts=None,
                    gt_track_tte=None):
        """Initializes tracking for the first frame of a new sequence.

        Args:
            det_feats (torch.Tensor, required): Feature representation 
                for the detections
            bbox (torch.Tensor, required): bounding boxes for each 
                detection
            lstm (nn.LSTM, required): lstm for processing tracks
            trackCount (int, required): the number of tracks currently
                allocated for this scene
            c0 (torch.Tensor): lstm initial cell state
            h0 (torch.Tensor): lstm initial hidden state
            sample_token (str, optional): token of the current detection
                used for computing performance metrics
            gt_bboxes (LiDARInstance3DBoxes): Object containing GT bounding boxes
            gt_tracks (torch.Tensor): GT tracks ids
            output_preds (bool): If True outputs tracking predictions o/w None
            return_loss (bool): If True compute the loss

        Returns: 
            trackCount (int): the new track count
            log_vars (dict): information to be logged 
            losses (dict): losses computed from the association matrix
            (torch.tensor) list of global tracking ids corresponding to the current 
                detections if output_preds is True, None otherwise
        """
        
        log_vars = {}
        losses = torch.tensor(0.,dtype=torch.float32,device=device)

        if det_feats.nelement() == 0:
            print('[WARNING] No detections for this frame',sample_token)
            #if detections were empty
            trk_id_preds = torch.tensor([],dtype=torch.float32,device=device)
            score_preds = torch.tensor([],dtype=torch.float32,device=device)
            bbox_preds = torch.tensor([],dtype=torch.float32,device=device)

            det_indices = torch.tensor([],dtype=torch.long,device=device)
            return trackCount, log_vars, losses, (trk_id_preds, score_preds, bbox_preds,)

        new_idx = np.arange(0,det_feats.size(0))
        self.incrementor(net=net,
                        active_tracks=self.activeTracks,
                        hidden_idx=[],
                        increment_hidden_idx=[],
                        new_idx=new_idx,
                        feats=det_feats,
                        device=device)
        
        track_feats = self.incrementor.gatherAny('f',new_idx)
        forecast_preds = net.MLPPredict(track_feats.clone())
        refine_preds = net.MLPRefine(track_feats.clone())
        refine_preds = torch.cat([refine_preds[:,:-1],torch.sigmoid(refine_preds[:,-1]).unsqueeze(1)],dim=1)


        self.tracks = [Track(id_=trackCount + ii, 
                            cls=self.cls, 
                            sample=sample_token,
                            det_confidence=det_confidence[ii],
                            bbox=x.clone(), 
                            score=refine_preds[ii,-1].detach().clone(),
                            futures=forecast_preds[ii,:],
                            xy=refine_preds[ii,-3:-1].detach().clone(),
                            timestep=timestep,) 
                        for ii,x in enumerate(bbox)]

        assert len(self.tracks) == self.incrementor.track_feats.size(0)

        trackCount += len(self.tracks)
        self.activeTracks = [i for i in range(len(self.tracks))]




        if gt_tracks is not None:
            if gt_bboxes == []:
                tp_det_idx, tp_gt_idx = [],[]
            else:
                ious, tp_det_idx, tp_gt_idx = self.get_iou_idx(bbox,gt_bboxes,device)
            # create a map from current tracks to their GT track number if their
            # associated detection is considered a TP
            self.trkid_to_gt_trkid = torch.full((len(self.tracks),),-1,dtype=torch.long,device=device)
            self.trkid_to_gt_tte = torch.full((len(self.tracks),),-1,dtype=torch.long,device=device)
            self.trkid_to_gt_trkid[tp_det_idx] = gt_tracks[tp_gt_idx]
            self.trkid_to_gt_tte[tp_det_idx] = gt_track_tte[tp_gt_idx]

            dets_to_trk_idx = torch.tensor(self.activeTracks,dtype=torch.long,device=device)
            refined_bboxes = torch.cat([self.tracks[i].refined_bboxes[-1].unsqueeze(0) for i in dets_to_trk_idx])


        #set only TP active 
        self.activeTracks = [i for i in tp_det_idx.cpu().numpy()]
        # print(tp_det_idx)
        # print(self.activeTracks)


        if return_loss and gt_tracks is not None: #only calculate loss information at training time
            refine_loss, summary, log = self.tsup.supervise_refinement(
                                            tp_det_idx=tp_det_idx,
                                            tp_gt_idx=tp_gt_idx,
                                            refine_preds=refine_preds,
                                            refined_bboxes=refined_bboxes,
                                            bbox=bbox,
                                            gt_pasts=gt_pasts,
                                            gt_bboxes=gt_bboxes,
                                            device=device,
                                            return_loss=return_loss)
            log_vars.update(summary)
            self.log_update(log)


            forecast_loss, summary, log = self.tsup.supervise_forecast(tp_det_idx=tp_det_idx,
                                                                        tp_gt_idx=tp_gt_idx,
                                                                        forecast_preds=forecast_preds,
                                                                        gt_futures=gt_futures,
                                                                        device=device,
                                                                        log_prefix='track_',
                                                                        return_loss=return_loss)
            log_vars.update(summary)
            self.log_update(log)

            losses = refine_loss + forecast_loss
        
        #get track outputs for this step
        # det_indices = torch.arange(0,len(self.tracks) ,dtype=torch.long,device=device)
        trk_id_preds, score_preds, bbox_preds, cls_preds = self.get_global_track_id_preds(self.activeTracks,device=device,output_preds=output_preds)
        return trackCount, log_vars, losses, (trk_id_preds, score_preds, bbox_preds, cls_preds,)