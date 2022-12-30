import numpy as np
import torch

import torch.nn as nn
from mmdet3d.models.builder import TRACKERS

from functools import reduce
from scipy.optimize import linear_sum_assignment
import copy


@TRACKERS.register_module()
class TrackingDecisionModifier():
    """
    Class designed to modify the decisions made by the tracker (i.e. for teacher forcing)
    """

    def __init__(self, teacher_forcing_mode='gt',decision_sampling={'det_newborn':'linear_decay',
                                                                    'det_false_positive':'linear_decay',
                                                                    'det_match':'linear_decay',
                                                                    'trk_match':'linear_decay'},):
        """
        decision_correct_sample_rate (dict): The rate of correct decisions for each decision type
        """
        super().__init__()

        self.decision_sampling = decision_sampling
        self.epoch = 0
        self.teacher_forcing_mode = teacher_forcing_mode

    def set_epoch(self,epoch,max_epoch):
        self.epoch = epoch
        self.max_epoch = max_epoch

    def det_false_positive(self,tp_det,num_dets,device,*args,**kwargs):
        mask = torch.zeros(num_dets,device=device)
        mask[tp_det] = 1
        return {'pos_det_false_positive': torch.where(mask == 0)[0],
                'neg_det_false_positive': tp_det}

    def det_newborn(self,tp_det,det_match,num_dets,device,*args,**kwargs):
        mask = torch.zeros(num_dets,device=device)
        mask[tp_det] = 1
        mask[det_match] = 0
        return {'pos_det_newborn': torch.where(mask == 1)[0],
                'neg_det_newborn': torch.where(mask == 0)[0]}

    def track_false_positive(self,tte_track,tp_track,num_tracks,device,*args,**kwargs):
        mask = torch.zeros(num_tracks,device=device)
        mask[reduce(torch.logical_or,[tp_track == -1,tte_track < 0])] = 1
        return {'pos_track_false_positive': torch.where(mask == 1)[0],
                'neg_track_false_positive': torch.where(mask == 0)[0]}
    
    def track_false_negative(self,tte_track,tp_track,tp_decisions,num_tracks,device,*args,**kwargs):
        mask = torch.zeros(num_tracks,device=device)
        mask[reduce(torch.logical_and,[tp_track != -1,tte_track >= 0])] = 1
        mask[tp_decisions['pos_track_match']] = 0
        return {'pos_track_false_negative': torch.where(mask == 1)[0],
                'neg_track_false_negative': torch.where(mask == 0)[0]}


    def __call__(self,tracker,bbox,gt_bboxes,gt_tracks,prev_AT,pred_cls,gt_labels,decisions,device):
        """Get the index of the GT bounding box that corresponds to the current detection"""
        num_dets = bbox.tensor.size(0)
        num_tracks = len(prev_AT)

        if gt_bboxes == []:
            tp_det_idx, tp_gt_idx = [],[]
        else:
            ious, tp_det_idx, tp_gt_idx = tracker.get_iou_idx(bbox,gt_bboxes,device,cls1=pred_cls,cls2=gt_labels)

        # Get mapping from active tracks to their GT trk ids
        active_trk_idx_to_prev_gt_trkid = tracker.trkid_to_gt_trkid[prev_AT] #breaks when test to train without restarting the sequence

        # Get mapping from current detections to their GT trk ids. -5 if false positive
        tp_det_to_gt_trkid = torch.full((num_dets,),-5,device=device)
        tp_det_to_gt_trkid[tp_det_idx] = gt_tracks[tp_gt_idx]

        # Match dets to tracks based on GT track id
        _, match_trk_pos_idx, match_det_pos_idx = np.intersect1d(
                    active_trk_idx_to_prev_gt_trkid.cpu().numpy(),
                    tp_det_to_gt_trkid.cpu().numpy(),
                    return_indices=True
                )
        match_trk_pos_idx, match_det_pos_idx = torch.from_numpy(match_trk_pos_idx).to(device), torch.from_numpy(match_det_pos_idx).to(device)

        # Create ground truth association matrix
        mask = torch.zeros(len(prev_AT),bbox.tensor.size(0),device=device)
        pos_matches = (match_trk_pos_idx,match_det_pos_idx,)
        mask[pos_matches] = 1.0
        match_trk_neg_idx, match_det_neg_idx = torch.where(mask == 0.)
            
        tp_decisions = {}
        tp_decisions['pos_track_match'] = match_trk_pos_idx
        tp_decisions['neg_track_match'] = match_trk_neg_idx
        tp_decisions['pos_det_match'] = match_det_pos_idx
        tp_decisions['neg_det_match'] = match_det_neg_idx

        if tracker.tracking_decisions == []:
            #workaround to handle the no tracking decisions case
            temp__ = torch.zeros(num_tracks,device=device)
            temp__[match_trk_pos_idx] = 1
            tp_decisions['pos_track_unmatched'] = torch.where(temp__ == 0)[0]
            

        for k in tracker.detection_decisions + tracker.tracking_decisions:
            tp_decisions.update(getattr(self,k)(tp_det=tp_det_idx,
                                                tp_track=active_trk_idx_to_prev_gt_trkid,
                                                tp_decisions=tp_decisions,
                                                tte_track=tracker.trkid_to_gt_tte[prev_AT],
                                                trk_match=match_trk_pos_idx,
                                                det_match=match_det_pos_idx,
                                                num_tracks=num_tracks,
                                                num_dets=num_dets,
                                                device=device))

        if tracker.teacher_forcing:
            mod_decisions = self.get_modified_decisions(tracker,
                                                        tp_decisions,
                                                        num_tracks=num_tracks,
                                                        num_dets=num_dets,
                                                        decisions=decisions,
                                                        device=device)
        else:
            mod_decisions = []

        self.get_stats(tracker,decisions,tp_decisions,device)
        
        return tp_decisions, mod_decisions, tp_det_idx, tp_gt_idx 


    def get_stats(self,tracker,decisions,tp_decisions,device):
        
        for k,v in decisions.items():
            # print('here')
            if k == 'track_match' or k == 'track_unmatched' or k == 'det_unmatched':
                continue
            # print('here1')
            if len(tp_decisions['pos_'+k]) == 0:
                continue

            # print('here2')

            gt = tp_decisions['pos_'+k].cpu().numpy()
            pred = decisions[k].cpu().numpy()
            # print('gt',gt)
            # print('pred',pred)
            if k == 'det_match':
                gt_trk = tp_decisions['pos_track_match'].cpu().numpy()
                pred_trk = decisions['track_match'].cpu().numpy()

                sgt = set([(gt_trk[i],gt[i],) for i in range(len(gt))])
                spred = set([(pred_trk[i],pred[i],) for i in range(len(pred))])
                first_idx = sgt.intersection(spred)
            else:
                _, first_idx, second_idx = np.intersect1d(pred,gt,return_indices=True)

             
            try:
                tracker.logging[k+'_gt'] += len(gt)
                tracker.logging[k+'_correct'] += len(first_idx)
                tracker.logging[k+'_num_pred'] += len(pred)

                tracker.logging['total_gt'] += len(gt)
                tracker.logging['total_correct'] += len(first_idx)
            except KeyError:
                tracker.logging[k+'_gt'] = len(gt)
                tracker.logging[k+'_correct'] = len(first_idx)
                tracker.logging[k+'_num_pred'] = len(pred)

                tracker.logging['total_gt'] = len(gt)
                tracker.logging['total_correct'] = len(first_idx)
        # print(tp_decisions)
        # print(decisions)
        # print(tracker.logging)


    def linear_decay(self):
        rate_dec = (self.epoch)/self.max_epoch
        return torch.tensor([1 - rate_dec, rate_dec])



    def sample_decision(self,decision,num_samples):
        try:
            # 0 means keep, while 1 means replace
            sample_rate = getattr(self,self.decision_sampling[decision])()
            choices = torch.multinomial(sample_rate, num_samples, replacement=True)
        except RuntimeError:
            print("[ERROR in sample_decision()]")
            print('num_samples',num_samples)
            print('decision',decision)
            exit(0)

        return choices


    def get_modified_decisions(self,tracker,tp_decisions,num_tracks,num_dets,decisions,device):
        """This methods implements schecduled sampling for teacher forcing"""
        if self.teacher_forcing_mode == 'gt':
            temp = dict()
            for k in decisions.keys():
                try:
                    temp[k] = tp_decisions['pos_'+k]
                except KeyError:
                    temp[k] = torch.tensor([],device=device)

            return temp

        dec_to_idx = {k:i for i,k in enumerate(decisions.keys())}
        track_to_dec = torch.full((num_tracks,),-1,device=device)
        track_to_dec_idx = torch.full((num_tracks,),-1,device=device)
        det_to_dec = torch.full((num_dets,),-1,device=device)
        det_to_dec_idx = torch.full((num_dets,),-1,device=device)
        for k,v in decisions.items():
            if k.startswith('track'):
                track_to_dec[v] = dec_to_idx[k]
                track_to_dec_idx[v] = torch.arange(0,len(v),device=device)
            elif k.startswith('det'):
                det_to_dec[v] = dec_to_idx[k]
                det_to_dec_idx[v] = torch.arange(0,len(v),device=device)
                

        track_mod_dec = torch.full((num_tracks,),-1,device=device)
        det_mod_dec = torch.full((num_dets,),-1,device=device)

        #decisions for matches first
        if len(tp_decisions['pos_det_match']) > 0:
            choices = self.sample_decision('det_match', len(tp_decisions['pos_det_match']))
            det_swap_idx = tp_decisions['pos_det_match'][choices==1]
            track_swap_idx = tp_decisions['pos_track_match'][choices==1]

            det_mod_dec[det_swap_idx] = det_to_dec[det_swap_idx]
            track_mod_dec[track_swap_idx] = track_to_dec[track_swap_idx]

        #detection decisions
        for k in tracker.detection_decisions:
            tp_dec = tp_decisions['pos_'+k]
            if len(tp_dec) == 0:
                continue
            
            choices = self.sample_decision(decision=k,num_samples=len(tp_dec))

            swap_idx = tp_decisions['pos_'+k][choices==1]
            det_mod_dec[swap_idx] = det_to_dec[swap_idx]

        #tracking decisions
        for k in tracker.tracking_decisions:
            tp_dec = tp_decisions['pos_'+k]
            if len(tp_dec) == 0:
                continue
            
            choices = self.sample_decision(decision=k,num_samples=len(tp_dec))

            swap_idx = tp_decisions['pos_'+k][choices==1]
            track_mod_dec[swap_idx] = track_mod_dec[swap_idx]

        
        # resolve initial match conflicts from PREDICTED decisions
        # det_match_idx = torch.where(det_mod_dec == dec_to_idx['det_match'])[0]
        # track_swap_idx = decisions['track_match'][det_to_dec_idx[det_match_idx]]
        # track_mod_dec[track_swap_idx] = track_to_dec[track_swap_idx]

        # track_match_idx = torch.where(track_mod_dec == dec_to_idx['track_match'])[0]
        # det_swap_idx = decisions['det_match'][track_to_dec_idx[track_match_idx]]
        # det_mod_dec[det_swap_idx] = det_to_dec[det_swap_idx]




        # print("before TP")
        # print('track_mod_dec',track_mod_dec)
        # print('det_mod_dec',det_mod_dec)
        # print('track_swap_idx',track_swap_idx)
        # print('det_swap_idx',det_swap_idx)
        # print()


        #create TP indexes
        track_to_dec_tp = torch.full((num_tracks,),-1,device=device)
        track_to_dec_idx_tp = torch.full((num_tracks,),-1,device=device)

        det_to_dec_tp = torch.full((num_dets,),-1,device=device)
        det_to_dec_idx_tp = torch.full((num_dets,),-1,device=device)

        for k,v in tp_decisions.items():
            if k.startswith('pos_track'):
                track_to_dec_tp[v] = dec_to_idx[k[4:]]
                track_to_dec_idx_tp[v] = torch.arange(0,len(v),device=device)
            elif k.startswith('pos_det'):
                det_to_dec_tp[v] = dec_to_idx[k[4:]]
                det_to_dec_idx_tp[v] = torch.arange(0,len(v),device=device)




        while(True):
            # resolve initial match conflicts from PREDICTED decisions
            det_match_idx = torch.where(det_mod_dec == dec_to_idx['det_match'])[0]
            track_swap_idx = decisions['track_match'][det_to_dec_idx[det_match_idx]]
            track_mod_dec[track_swap_idx] = track_to_dec[track_swap_idx]

            track_match_idx = torch.where(track_mod_dec == dec_to_idx['track_match'])[0]
            det_swap_idx = decisions['det_match'][track_to_dec_idx[track_match_idx]]
            det_mod_dec[det_swap_idx] = det_to_dec[det_swap_idx]

            # where mod decisions are det match
            det_conflict_idx = torch.where(det_mod_dec == dec_to_idx['det_match'])[0]
            # where TP decisions are also det match 
            det_conflict_idx_tp = torch.where(det_to_dec_tp[det_conflict_idx] == dec_to_idx['det_match'])[0]
            if len(det_conflict_idx_tp) > 0:
                # the TP track indices corresponding to det_conflict_idx_tp
                track_conflict_idx_tp = tp_decisions['pos_track_match'][det_to_dec_idx_tp[det_conflict_idx[det_conflict_idx_tp]]]
                det_remain = torch.where(track_mod_dec[track_conflict_idx_tp] == -1 )[0]
                if len(det_remain) != 0:
                    track_mod_dec[track_conflict_idx_tp[det_remain]] = track_to_dec[track_conflict_idx_tp[det_remain]]
            else:
                det_remain = []


            track_conflict_idx = torch.where(track_mod_dec == dec_to_idx['track_match'])[0]
             # where TP decisions are also det match 
            track_conflict_idx_tp = torch.where(track_to_dec_tp[track_conflict_idx] == dec_to_idx['track_match'])[0]

            if len(track_conflict_idx_tp) > 0:
                det_conflict_idx_tp = tp_decisions['pos_det_match'][track_to_dec_idx_tp[track_conflict_idx[track_conflict_idx_tp]]]
                track_remain = torch.where( det_mod_dec[det_conflict_idx_tp] == -1 )[0]
                if len(track_remain) != 0:
                    det_mod_dec[det_conflict_idx_tp[track_remain]] = det_to_dec[det_conflict_idx_tp[track_remain]]
            else:
                track_remain = []

            if len(track_remain) == 0 and len(det_remain) == 0:
                break



        
        # fill in the remaining entries with TP decisions
        track_fill_idx = torch.where(track_mod_dec == -1)[0]
        track_mod_dec[track_fill_idx] = track_to_dec_tp[track_fill_idx]
        # print('track_to_dec_tp',track_to_dec_tp)
        # print('track_mod_dec',track_mod_dec)
        # print('track_fill_idx',track_fill_idx)

        det_fill_idx = torch.where(det_mod_dec == -1)[0]
        det_mod_dec[det_fill_idx] = det_to_dec_tp[det_fill_idx]
        # print('det_to_dec_tp',det_to_dec_tp)
        # print('det_mod_dec',det_mod_dec)
        # print('det_fill_idx',det_fill_idx)

        # resolve match conflicts for TP decisions
        track_match_idx = torch.where(track_mod_dec[track_fill_idx] == dec_to_idx['track_match'])[0]
        det_swap_idx = tp_decisions['pos_det_match'][track_to_dec_idx_tp[track_fill_idx][track_match_idx]]
        det_mod_dec[det_swap_idx] = det_to_dec_tp[det_swap_idx]


        det_match_idx = torch.where(det_mod_dec[det_fill_idx] == dec_to_idx['det_match'])[0]
        track_swap_idx = tp_decisions['pos_track_match'][det_to_dec_idx_tp[det_fill_idx][det_match_idx]]
        track_mod_dec[track_swap_idx] = track_to_dec_tp[track_swap_idx]


        match_idx_track_ = torch.where(track_mod_dec == dec_to_idx['track_match'])[0]
        match_idx_det_ = torch.where(det_mod_dec == dec_to_idx['det_match'])[0]
        if len(match_idx_track_) < len(match_idx_det_):
            pass
        elif len(match_idx_det_) < len(match_idx_track_):
            pass



        #fill modified decisions
        mod_decisions = {}
        for k,i in dec_to_idx.items():
            if k.startswith('track'):
                mod_decisions[k] = torch.where(track_mod_dec == i)[0]
            elif k.startswith('det'):
                mod_decisions[k] = torch.where(det_mod_dec == i)[0]



        

        try:
            assert len(mod_decisions['det_match']) == len(mod_decisions['track_match'])
        except AssertionError:
            print("[AssertionError] assert len(mod_decisions['det_match']) == len(mod_decisions['track_match'])")
            print('num_dets',num_dets)
            print('num_tracks',num_tracks)
            print('      mod_decisions:',[[k,x.size(0)] for k,x in mod_decisions.items()],sum([x.size(0) for k,x in mod_decisions.items()]))
            print('mod_decisions det  : ',sum([x.size(0) for k,x in mod_decisions.items()  if k[:3] == 'det']) )
            print('mod_decisions track: ',sum([x.size(0) for k,x in mod_decisions.items() if k[:5] == 'track']) )
            print('      decisions    :',[[k,x.size(0)] for k,x in decisions.items()],sum([x.size(0) for k,x in decisions.items()]))
            print('decisions det      : ',sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) )
            print('decisions track    : ',sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']) )
            print('      tp_decisions :',[[k,x.size(0)] for k,x in tp_decisions.items() if k[:4] == 'pos_'],sum([x.size(0) for k,x in tp_decisions.items() if k[:4] == 'pos_']))
            print('tp_decisions det   : ',sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) )
            print('tp_decisions track : ',sum([x.size(0) for k,x in tp_decisions.items() if k[:9] == 'pos_track']) )
            print()
            print()
            print('track_mod_dec',track_mod_dec)
            print('det_mod_dec',det_mod_dec)
            print('dec_to_idx',dec_to_idx)
            print('track_match_idx',track_match_idx)
            print('det_match_idx',det_match_idx)
            print()
            exit(0)


        try:
            assert sum([x.size(0) for x in mod_decisions.values()]) == sum([x.size(0) for x in decisions.values()])
        except AssertionError:
            print("[AssertionError] assert sum([x.size(0) for x in mod_decisions.values()]) == sum([x.size(0) for x in decisions.values()])")
            print('num_dets',num_dets)
            print('num_tracks',num_tracks)
            print('      mod_decisions:',[[k,x.size(0)] for k,x in mod_decisions.items()],sum([x.size(0) for k,x in mod_decisions.items()]))
            print('mod_decisions det  : ',sum([x.size(0) for k,x in mod_decisions.items()  if k[:3] == 'det']) )
            print('mod_decisions track: ',sum([x.size(0) for k,x in mod_decisions.items() if k[:5] == 'track']) )
            print('      decisions:',[[k,x.size(0)] for k,x in decisions.items()],sum([x.size(0) for k,x in decisions.items()]))
            print('decisions det  : ',sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) )
            print('decisions track: ',sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']) )
            print('      tp_decisions:',[[k,x.size(0)] for k,x in tp_decisions.items() if k[:4] == 'pos_'],sum([x.size(0) for k,x in tp_decisions.items() if k[:4] == 'pos_']))
            print('tp_decisions det  : ',sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) )
            print('tp_decisions track: ',sum([x.size(0) for k,x in tp_decisions.items() if k[:9] == 'pos_track']) )
            print()
            print('track_mod_dec',track_mod_dec)
            print('det_mod_dec',det_mod_dec)
            print('dec_to_idx',dec_to_idx)
            print()
            exit(0)
        except AttributeError:
            print(mod_decisions)
            print(decisions)

        assert sum([x.size(0) for k,x in mod_decisions.items()  if k[:3] == 'det']) == sum([x.size(0) for k,x in decisions.items() if k[:3] == 'det'])
        assert sum([x.size(0) for k,x in mod_decisions.items()  if k[:5] == 'track']) == sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track'])

        return mod_decisions
        




        

    

    

