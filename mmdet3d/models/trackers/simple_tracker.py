
import torch

import torch.nn as nn
import numpy as np

from scipy.optimize import linear_sum_assignment
from mmdet3d.core.bbox.iou_calculators import BboxOverlapsNearest3D
from .track import Track
from .tracking_helpers import getTPS

from mmdet3d.models import TRACKERS
from mmdet3d.models  import builder


@TRACKERS.register_module()
class SimpleTracker(nn.Module):
    ioucal = BboxOverlapsNearest3D('lidar')

    def __init__(self, cls, margin=10, trackDistanceThreshold=30, frameLimit=10, tp_threshold=0.6):
        """The Track Manager implements the logic for tracking objects of the same class

        args:
            cls (string): class name
            trackDistanceThreshold (float): threshold on the MLPPair distance between two
                detection and track pairs. tracks are only matched if their score is small enough
            frameLimit (int): The number of frames to persist unmatched tracks 
            tp_threshold (float): tracking score threshold for true positives
        
        """
        super().__init__()
        # super(TrackManager, self).__init__()
        self.cls = cls
        self.tracks = []

        self.activeTracks = [] # tensor index : track index
        self.unmatchedTracks = {} # tensor index : track index
        self.decomTracks = [] # tensor index : track index

        self.trackDistanceThreshold = trackDistanceThreshold
        self.frameLimit = frameLimit
        self.tp_threshold = tp_threshold
        self.trkid_to_gt_trkid = None

        self.marginLoss = nn.MarginRankingLoss(margin=margin)
        self.smoothL1 = nn.SmoothL1Loss()

    def reset(self):
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
        self.trkid_to_gt_trkid = None
        try:
            del self.prev_c
            del self.prev_h
            del self.track_feats
        except AttributeError:
            pass

    def gatherActive(self,feats):
        """Indexes into the appropriate tensor to gather the desired
        freatures."""
        out = []
        index = self.activeTracks
        for x in feats:
            if x == 'c':
                out.append(self.prev_c[index,:])
            elif x == 'h':
                out.append(self.prev_h[index,:])
            elif x == 'f':
                out.append(self.track_feats[index,:])
            else:
                print('invalid char must be in [c,h,f]')
        if len(out) > 1: 
            return (*out,)
        else:
            return out[0]
    
    def initTracks(self,det_feats,bbox,lstm,trackCount,c0,h0,device,sampleTok=None,
                   gt_bboxes=[],gt_tracks=None,output_preds=False,
                   return_loss=True):
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
            sampleTok (str, optional): token of the current detection
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
        
        self.tracks = [Track(trackCount + ii, self.cls, sampleTok, x) 
                        for ii,x in enumerate(bbox)]

        self.track_feats, (self.prev_h, self.prev_c) = lstm(
                        det_feats.unsqueeze(0), 
                        (h0.weight.unsqueeze(0).repeat(1,det_feats.size(0),1),
                         c0.weight.unsqueeze(0).repeat(1,det_feats.size(0),1))
                       )

        self.track_feats = self.track_feats.squeeze(0).clone()#clone is needed here
        self.prev_h = self.prev_h.squeeze(0).clone() # TODO check if this clone is actually needed
        self.prev_c = self.prev_c.squeeze(0).clone() #TODO check if this clone is actually needed
        num_tracks = self.track_feats.size(0)

        trackCount += num_tracks
        self.activeTracks = [i for i in range(num_tracks)]


        if return_loss: #only calculate loss information at training time
            
            if gt_bboxes == []:
                tp_det_idx, tp_gt_idx = [],[]
            else:
                tp_det_idx, tp_gt_idx = getTPS(bbox.tensor,gt_bboxes.tensor,
                        threshold=self.tp_threshold,ioucal=SimpleTracker.ioucal
                    )

            #create a map from current tracks to their GT track number if the 
            # associated detection is considered a TP
            self.trkid_to_gt_trkid = torch.full((len(self.tracks),),-1,device=device)
            self.trkid_to_gt_trkid[tp_det_idx] = gt_tracks[tp_gt_idx]
        
        
        
        return trackCount, log_vars, {'l1':0,'margin':0}, self.get_global_track_id_preds(self.activeTracks, output_preds=output_preds)
        
    def step(self,det_feats,bbox,lstm,trackCount,MLPPair,c0,h0,device,last_in_scene,sampleTok=None,
             gt_bboxes=[],gt_tracks=None,output_preds=False,return_loss=True,):
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
            sampleTok (str, optional): token of the current detection
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
        if self.tracks == []: #when tracks for this class were not initialized
            trackCount, log_vars, loss, track_pred = self.initTracks(
                det_feats,bbox,lstm,
                trackCount,c0,h0,device,
                sampleTok,gt_bboxes=gt_bboxes,gt_tracks=gt_tracks,
                return_loss=return_loss,output_preds=output_preds)
            return trackCount, log_vars, loss, track_pred
        log_vars = {}
        losses = {}

        #START -- track-detection association process
        trk_feats = self.gatherActive(['f'])
        trk_feats = trk_feats.clone()
        
        prod = torch.cartesian_prod(
            torch.arange(0,trk_feats.size(0),dtype=torch.long),
            torch.arange(0,det_feats.size(0),dtype=torch.long)
        )
        cat = torch.cat([trk_feats[prod[:,0]],det_feats[prod[:,1]]],dim=1)
        dists = MLPPair(cat)

        assert dists.size(1) == 1

        cost_mat = torch.zeros(trk_feats.size(0), det_feats.size(0),
                               dtype=cat.dtype, device=cat.device)
        cost_mat[prod[:,0],prod[:,1]] = dists.squeeze(1)

        log_vars[f"{self.cls}_min_cost"] = torch.min(cost_mat.clone().detach()).item()
        log_vars[f"{self.cls}_mean_cost"] = torch.mean(cost_mat.clone().detach()).item()
        
        if return_loss:
            cost_mat.retain_grad()

        trk_ind, det_ind = linear_sum_assignment(cost_mat.clone().detach().cpu().numpy())
        #END -- track-detection association process


        #START -- cost thresholding 
        costs = cost_mat[trk_ind, det_ind]
        np_costs = costs.detach().cpu().numpy()
        match_idx = np.where(np_costs < self.trackDistanceThreshold)
        no_match_idx = np.where(np_costs >= self.trackDistanceThreshold)
        trk_match = trk_ind[match_idx]
        det_match = det_ind[match_idx]

        if trk_feats.size(0) > det_feats.size(0): 
            #more tracks than dets
            #need to find unmatched
            trk_unmatched = set([x for x in range(trk_feats.size(0))]).difference(trk_match)
            trk_new = det_ind[no_match_idx]
        elif trk_feats.size(0) <= det_feats.size(0): 
            #more dets than tracks
            #no need to find unmatched
            trk_unmatched = trk_ind[no_match_idx]
            trk_new = set([x for x in range(det_feats.size(0))]).difference(det_match)
        else: 
            print('Should not get here')
            raise Exception

        
        tempTensorToTrack  = {i:v for i,v in enumerate(self.activeTracks)}
        at_tensor = torch.tensor(self.activeTracks,dtype=torch.long)
        dets_to_trk_idx = torch.zeros(det_feats.size(0),dtype=torch.long)
        assert (len(det_match) + len(trk_new)) == det_feats.size(0)
        # print(self.activeTracks)
        dets_to_trk_idx[det_match] = at_tensor[trk_match]
        # print(trk_new)
        dets_to_trk_idx[np.array(list(trk_new))] = torch.tensor([len(self.tracks) + ii for ii in range(len(trk_new))],
                                                                dtype=torch.long)
        #END -- cost thresholding

        
        
        #START LSTM step & update tracking buffers based on decisions
        h_trk = self.gatherActive(['h'])[trk_match].unsqueeze(0)
        c_trk = self.gatherActive(['c'])[trk_match].unsqueeze(0)
        feats_in = det_feats[det_match].unsqueeze(0)

        activeTrackUpdate = []
        new_trk_count = len(trk_new)
        if new_trk_count != 0:
            #add any new tracks by concatenating their corresponding tensors
            #to the input for the LSTM step
            trk_new = list(trk_new)
            self.tracks += [Track(trackCount + ii, self.cls, sampleTok, x) 
                            for ii,x in enumerate(bbox[trk_new])]
            
            activeTrackUpdate += [len(self.activeTracks)+i for i in range(new_trk_count)]
            trackCount += len(trk_new)


            if not last_in_scene:
                h_new = h0.weight.unsqueeze(0).repeat(1,new_trk_count,1)
                h_trk = torch.cat([h_trk,h_new],dim=1)
                c_new = c0.weight.unsqueeze(0).repeat(1,new_trk_count,1)
                c_trk = torch.cat([c_trk,c_new],dim=1)
                feats_new = det_feats[trk_new].unsqueeze(0)
                feats_in = torch.cat([feats_in,feats_new],dim=1)

        if not last_in_scene:
            feats_in = feats_in.clone()
            h_trk = h_trk.clone()
            c_trk = c_trk.clone()


            out_trk, (out_h, out_c) = lstm(feats_in,(h_trk,c_trk))

            if new_trk_count != 0:
                self.track_feats = torch.cat([self.track_feats,out_trk[0,-new_trk_count:,:]],dim=0).clone()
                self.prev_h = torch.cat([self.prev_h,out_h[0,-new_trk_count:,:]],dim=0).clone()
                self.prev_c = torch.cat([self.prev_c,out_c[0,-new_trk_count:,:]],dim=0).clone()

            prev_trk_update_idx = torch.tensor(self.activeTracks,dtype=torch.long)[trk_match]
            self.track_feats[prev_trk_update_idx] = out_trk[:,:len(trk_match),:].clone()
            self.prev_h[prev_trk_update_idx] = out_h[:,:len(trk_match),:].clone()
            self.prev_c[prev_trk_update_idx] = out_c[:,:len(trk_match),:].clone()

            self.track_feats = self.track_feats.clone()
            self.prev_h = self.prev_h.clone()
            self.prev_c = self.prev_c.clone()

            #update existing tracks
            for deti,trki in enumerate(trk_match):
                idx = tempTensorToTrack[trki]
                    
                self.tracks[idx].addTimestep(
                                sample=sampleTok,
                                bbox=bbox.tensor[det_match[deti],:]
                            )
                activeTrackUpdate.append(idx)

            decommishionnedTrackUpdate = []
            for i in trk_unmatched:
                idx = tempTensorToTrack[i]
                self.tracks[idx].unmatched_step()
                if self.tracks[idx].unmatchedFrameCount > self.frameLimit:
                    decommishionnedTrackUpdate.append(idx)
                else:
                    activeTrackUpdate.append(idx)

            self.activeTracks = activeTrackUpdate
            self.decomTracks += decommishionnedTrackUpdate
            #END LSTM step & update tracking buffers based on decisions
        else:
            # print('skipped LSTM step at last iter')
            pass



        #START SUPERVISE
        
        if return_loss: #only calculate loss info during training

            if gt_bboxes == []:
                tp_det_idx, tp_gt_idx = [],[]
            else:
                tp_det_idx, tp_gt_idx = getTPS(bbox.tensor,gt_bboxes.tensor,
                        threshold=self.tp_threshold,ioucal=SimpleTracker.ioucal
                    )
        

            #active track idx mapped to their GT trk ids from T-1
            active_trk_prev_gt_trkid = self.trkid_to_gt_trkid[list(tempTensorToTrack.values())]#breaks when test to train without restarting the sequence
            #tp det list of GT trk ids
            tp_det_to_gt_trkid = torch.full_like(dets_to_trk_idx,-5,device=device)
            try:
                tp_det_to_gt_trkid[tp_det_idx] = gt_tracks[tp_gt_idx]
            except IndexError:
                    print('tp_det_to_gt_trkid',tp_det_to_gt_trkid)
                    print('tp_det_idx',tp_det_idx)
                    print('gt_tracks',gt_tracks)
                    print('tp_gt_idx',tp_gt_idx)

            _, trk_pos_sup_idx, det_pos_sup_idx = np.intersect1d(
                        active_trk_prev_gt_trkid.cpu().numpy(),
                        tp_det_to_gt_trkid.cpu().numpy(),
                        return_indices=True
                    )


            #create mask to efficiently index negative entries 
            mask = torch.zeros_like(cost_mat)

            log_vars[f"{self.cls}_num_pos_sup"] = len(trk_pos_sup_idx)

            #Supervise correct (Positive)
            if len(trk_pos_sup_idx) == 0:
                smoothl1_loss = torch.tensor(0.,device=device, requires_grad=True)
            else:
                # pos_matches = (np.array(tp_trk_match_idx_T[pos_sup_idx]),np.array(tp_det_match_idx_T[pos_sup_idx]),)
                pos_matches = (trk_pos_sup_idx,det_pos_sup_idx,)
                mask[pos_matches] = 1
                pos_sup_targets = cost_mat[pos_matches].flatten()
                smoothl1_loss = self.smoothL1(pos_sup_targets,torch.full_like(pos_sup_targets,0.))
            
            #Supervise incorrect (Negative)
            where_0 = torch.where(mask == 0)
            if where_0[0].numel() == 0:
                margin_loss = torch.tensor(0.,device=device, requires_grad=True)
            else:
                neg_sup_targets = cost_mat[where_0].flatten()
                neg_sup_targets.retain_grad()

                # this will incourage the first input to be larger than the margin o/w if incurs nonzero loss
                margin_loss = self.marginLoss(neg_sup_targets,
                                              torch.full_like(neg_sup_targets,0.),
                                              torch.full_like(neg_sup_targets,1.))

                

            losses.update({'l1':smoothl1_loss,'margin':margin_loss,'cost_mat':cost_mat})
            if losses['margin'].isnan():
                print('losses l1:',losses['l1'])
                print('margin loss:',losses['margin'])
                print('[NAN] cost_mat.shape() (trk x dets)',cost_mat.shape,cost_mat)


            #update the ground truth track mapping to prevent penalizing id switches again and again
            update_ = torch.full((len(self.tracks),),-1,device=device)
            update_[:self.trkid_to_gt_trkid.size(0)] = self.trkid_to_gt_trkid
            
            assert dets_to_trk_idx.size(0) == bbox.tensor.size(0)
            update_[dets_to_trk_idx[tp_det_idx]] = gt_tracks[tp_gt_idx]

            self.trkid_to_gt_trkid = update_
            # print(self.trkid_to_gt_trkid.shape)
        #END SUPERVISE
        
        return trackCount, log_vars, losses, self.get_global_track_id_preds(dets_to_trk_idx, output_preds=output_preds)


    def get_global_track_id_preds(self, dets_to_trk_idx, output_preds=True):
        """
        Args:
            dets_to_trk_idx (torch.Tensor): list of class-specific track
                ids corresponding to the current detections
            output_preds (bool): whether to compute predictions or not 
        
        Returns:
            (torch.tensor) list of global tracking ids corresponding to the current 
                detections if predicte is True, None otherwise
        """

        if output_preds:
            return torch.tensor([self.tracks[i].id for i in dets_to_trk_idx])
        else:
            return None




