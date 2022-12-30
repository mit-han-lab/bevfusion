import numpy as np
import torch

import torch.nn as nn
from mmdet3d.models.builder import TRACKERS

from functools import reduce
from scipy.optimize import linear_sum_assignment


@TRACKERS.register_module()
class TrackingAssociator():

    def __init__(self,cost_mat_type='margin',use_distance_prior=False):
        super().__init__()
        self.use_distance_prior = use_distance_prior
        self.cost_mat_type = cost_mat_type
        self.row_smx = nn.Softmax(dim=1)
        self.col_smx = nn.Softmax(dim=0)


    def get_cost_mat_margin(self,num_det,num_trk,supervise,tracker,cost_mat,track_det_dists,device):
        cost_mat[:num_trk,:num_det] = supervise['cost_mat']
        
        # add extra distance cost
        if track_det_dists != [] and self.use_distance_prior:
            temp_ = torch.full_like(cost_mat[:num_trk,:num_det],0)
            # print(track_det_dists.shape,temp_.shape)
            # 99.99% of nusc cars travel slower than 22m/s
            temp_[torch.where(track_det_dists > 22)] = 3
            cost_mat[:num_trk,:num_det] += temp_

        if len(tracker.tracking_decisions) > 0:
            cm_temp = cost_mat[:num_trk,:num_det].clone()
            for d in range(len(tracker.detection_decisions)):
                for t in range(len(tracker.tracking_decisions)):
                    cost_mat[num_trk + d * num_det:num_trk + (d + 1) * num_det,
                            num_det + t * num_trk:num_det + (t + 1) * num_trk] = cm_temp.T


        #update detection decisions
        col = torch.arange(0,num_det,device=device)
        for i,k in enumerate(tracker.detection_decisions):
            row = col + num_trk + num_det * i
            cost_mat[row,col] = supervise[k]

        #update tracking decisions
        row = torch.arange(0,num_trk,device=device)
        for i,k in enumerate(tracker.tracking_decisions):
            col = row + num_det + num_trk * i
            cost_mat[row,col] = supervise[k]

        return cost_mat


    def get_cost_mat_softmax(self,num_det,num_trk,supervise,tracker,cost_mat,track_det_dists,device):
        decision_smx_cost = {}

        dd_mat = self.col_smx(
            torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(0) for k in tracker.detection_decisions ],dim=0)
        )
        decision_smx_cost.update({k:dd_mat[num_trk+i,:] for i,k in enumerate(tracker.detection_decisions)})

        td_mat = self.row_smx(
            torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(1) for k in tracker.tracking_decisions ],dim=1)
        )
        decision_smx_cost.update({k:td_mat[:,num_det+i] for i,k in enumerate(tracker.tracking_decisions)})


        if num_det > 0 and num_trk > 0:
            # as in deep association networks
            cost_mat[:num_trk,:num_det] = torch.max(
                torch.cat([td_mat[:num_trk,:num_det].unsqueeze(0),dd_mat[:num_trk,:num_det].unsqueeze(0)],dim=0),dim=0
            ).values * -1.0 # for hungarian algorithm

            if len(tracker.tracking_decisions) > 0:
                cm_temp = cost_mat[:num_trk,:num_det].clone()
                for d in range(len(tracker.detection_decisions)):
                    for t in range(len(tracker.tracking_decisions)):
                        cost_mat[num_trk + d * num_det : num_trk + (d + 1) * num_det,
                                num_det + t * num_trk : num_det + (t + 1) * num_trk] = cm_temp.T

        if num_det > 0:
            #update detection decisions
            col = torch.arange(0,num_det,device=device)
            for i,k in enumerate(tracker.detection_decisions):
                row = col + num_trk + num_det * i
                cost_mat[row,col] = decision_smx_cost[k] * -1.0 # for hungarian algorithm
            

        if num_trk > 0:
            #update tracking decisions
            row = torch.arange(0,num_trk,device=device)
            for i,k in enumerate(tracker.tracking_decisions):
                col = row + num_det + num_trk * i
                cost_mat[row,col] = decision_smx_cost[k] * -1.0 # for hungarian algorithm

        return cost_mat 

    def get_cost_mat(self, *args, **kwargs):
        return getattr(self,"get_cost_mat_{}".format(self.cost_mat_type))(*args, **kwargs)
    



    def __call__(self, supervise, tracker, num_trk, num_det, class_name, track_det_dists, class_mask, device):
        """Method which applies hungarian matching to the detection and tracks

        Args:
            net (): network containing all differentiable components 
            class_name (): the names of the current class  
            trk_feats (torch.Tensor): the current track features
            det_feats (torch.Tensor): the current det features
            track_det_dists (torch.Tensor): distances betten the tracks and detections in the prev frame.
            device (torch.devics): the device to use

        Returns:
            trk_unmatched (list): the indices of active tracks which were not matched
            trk_new (list): the indices of current detections which were not matched
            det_match (list): the indices of current detections which were matched
            trk_match (list): the indices of active tracks which were matched
        """
        summary = {}

        # assert dists.size(1) == 1
        cost_mat = torch.full((num_trk + tracker.dd_num * num_det, num_det + tracker.td_num * num_trk,),
                              10000.0,
                              dtype=torch.float32,
                              device=device)

        with torch.no_grad():
            cost_mat = self.get_cost_mat(num_det=num_det,
                                        num_trk=num_trk,
                                        supervise=supervise,
                                        tracker=tracker,
                                        cost_mat=cost_mat,
                                        track_det_dists=track_det_dists,
                                        device=device)

        
        trk_ind, det_ind = linear_sum_assignment(cost_mat.detach().clone().cpu().numpy())
        trk_ind = torch.from_numpy(trk_ind).to(device)
        det_ind = torch.from_numpy(det_ind).to(device)

        # prune out the extra decisions
        incorrect_idx = torch.where(cost_mat[trk_ind,det_ind] == 10000.0)[0]
        if len(incorrect_idx) > 0:
            print("Error len(use_idx) > 0")
            exit(0)
        # trk_ind = trk_ind[use_idx]
        # det_ind = det_ind[use_idx]

        if len(tracker.tracking_decisions) > 0:
            #enfoce that there be a tracking decision for each track
            use_idx = torch.zeros(len(trk_ind),dtype=torch.bool,device=device)
            use_idx[torch.where(reduce(torch.logical_and,[num_trk <= trk_ind, num_det <= det_ind]))] = 1
            use_idx = torch.where(use_idx == 0)[0]
            trk_ind = trk_ind[use_idx]
            det_ind = det_ind[use_idx]


            #if some tracks and detections have been forgotten, fix that
            temp_trk = torch.zeros(cost_mat.size(0),device=device)
            temp_det = torch.zeros(cost_mat.size(1),device=device)

            temp_trk[trk_ind] = 1
            unmatched_trk = torch.where(temp_trk[:num_trk] == 0)[0]

            cost_mat_copy = cost_mat.clone()

            if len(unmatched_trk) > 0:
                # print('inside len(unmatched_trk)>0',unmatched_trk,num_trk)
                temp = cost_mat_copy[unmatched_trk,:]
                if len(temp.shape) == 1:
                    temp = temp.unsqueeze(0)
                temp[:,det_ind] = 10000.0
                add_det_idx = torch.min(temp,axis=1).indices
                # print(temp.shape,temp)
                # print('add_det_idx',add_det_idx)
                trk_ind = torch.cat([trk_ind,unmatched_trk])
                det_ind = torch.cat([det_ind,add_det_idx])

            temp_det[det_ind] = 1
            unmatched_det = torch.where(temp_det[:num_det] == 0)[0]

            if len(unmatched_det) > 0: 

                # print('inside len(unmatched_det)>0',unmatched_det)
                temp = cost_mat_copy[:,unmatched_det]
                if len(temp.shape) == 1:
                    temp = temp.unsqueeze(1)
                temp[trk_ind,:] = 10000.0
                add_trk_idx = torch.min(temp,axis=0).indices
                # print(temp.shape,temp)
                # print('add_trk_idx',add_trk_idx)
                trk_ind = torch.cat([trk_ind,add_trk_idx])
                det_ind = torch.cat([det_ind,unmatched_det])




        
        matches = torch.where(reduce(torch.logical_and,[trk_ind < num_trk, det_ind < num_det]))
        decisions = {}
        decisions['det_match'] = det_ind[matches]
        decisions['track_match'] = trk_ind[matches]
        
        for i,k in enumerate(tracker.detection_decisions):
            lower = num_trk + num_det * i
            upper = num_trk + num_det * ( i + 1 )
            # print("i:{},k:{},upper:{},lower:{},trk:{},det:{}".format(i,k,upper,lower,num_trk,num_det))
            decisions[k] = det_ind[
                torch.where(reduce(torch.logical_and,[lower <= trk_ind, trk_ind < upper]))
            ]

        for i,k in enumerate(tracker.tracking_decisions):
            lower = num_det + num_trk * i
            upper = num_det + num_trk * ( i + 1 )
            decisions[k] = trk_ind[
                torch.where(reduce(torch.logical_and,[lower <= det_ind, det_ind < upper]))
            ]


        
        if num_trk > sum( [ len(decisions['track_match']) ] + [ len(decisions[k]) for k in tracker.tracking_decisions ]):
            # print('in unmatched tracks')
            #unmatched tracks
            temp = torch.zeros(num_trk)
            temp[
                torch.cat([decisions['track_match']] + [decisions[k] for k in tracker.tracking_decisions])
            ] = 1
            decisions['track_unmatched'] = torch.where(temp == 0)[0]

        else:
            decisions['track_unmatched'] = torch.tensor([],dtype=torch.long,device=device)
        
        if num_det >  sum( [ len(decisions['det_match']) ] + [ len(decisions[k]) for k in tracker.detection_decisions ]):
            #unmatched detections
            temp = torch.zeros(num_det)
            temp[
                torch.cat([decisions['det_match']] + [decisions[k] for k in tracker.detection_decisions])
            ] = 1
            decisions['det_unmatched'] = torch.where(temp == 0)[0]
        else:
            decisions['det_unmatched'] = torch.tensor([],dtype=torch.long,device=device)

        summary.update(self.create_summary(tracker,supervise,class_name))

        try:
            assert num_det == sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) , "Det error"
            assert num_trk == sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']), 'track error'
        except AssertionError as e:
            print(e)
            print("[AssertionError] assert num_det == sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) ")
            print('or')
            print("[AssertionError] num_trk == sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track'])  ")
            print('      decisions:',[[k,x.size(0)] for k,x in decisions.items()],sum([x.size(0) for k,x in decisions.items()]))
            print('decisions det  : ',sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) )
            print('decisions track: ',sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']) )
            print('num_det',num_det)
            print('num_trk',num_trk)
            print('len(incorrect_idx)',len(incorrect_idx))
            print("len(trk_ind)",len(trk_ind))
            print("len(det_ind)",len(det_ind))

            print("sum( [ len(decisions['track_match']) ] + [ len(decisions[k]) for k in tracker.tracking_decisions ])",sum( [ len(decisions['track_match']) ] + [ len(decisions[k]) for k in tracker.tracking_decisions ]))
            exit(0)

            

        return decisions, summary, (None, None, None)
        


    def create_summary(self,tracker,supervision,class_name):
        summary = {}
        for k in tracker.detection_decisions:
            if supervision[k].nelement() == 0:
                continue
            summary["{}_mean_cost_{}".format(class_name,k)] = torch.mean(supervision[k]).item()
            summary["{}_min_cost_{}".format(class_name,k)] = torch.min(supervision[k]).item()
            summary["{}_median_cost_{}".format(class_name,k)] = torch.median(supervision[k]).item()

        for k in tracker.tracking_decisions:
            if supervision[k].nelement() == 0:
                continue
            summary["{}_mean_cost_{}".format(class_name,k)] = torch.mean(supervision[k]).item()
            summary["{}_min_cost_{}".format(class_name,k)] = torch.min(supervision[k]).item()
            summary["{}_median_cost_{}".format(class_name,k)] = torch.median(supervision[k]).item()

        for k in ['match']:
            if supervision['cost_mat'].nelement() == 0:
                continue
            summary["{}_mean_cost_{}".format(class_name,k)] = torch.mean(supervision['cost_mat']).item()
            summary["{}_min_cost_{}".format(class_name,k)] = torch.min(supervision['cost_mat']).item()
            summary["{}_median_cost_{}".format(class_name,k)] = torch.median(supervision['cost_mat']).item()

        return summary









@TRACKERS.register_module()
class TrackingAssociatorMax():

    def __init__(self,cost_mat_type='margin',use_distance_prior=False):
        super().__init__()
        self.use_distance_prior = use_distance_prior
        self.cost_mat_type = cost_mat_type
        self.row_smx = nn.Softmax(dim=1)
        self.col_smx = nn.Softmax(dim=0)


    def get_cost_mat_margin(self,num_det,num_trk,supervise,tracker,cost_mat,track_det_dists,device):

        if num_det > 0 and num_trk > 0:
            cost_mat[:num_trk,:num_det] = supervise['cost_mat']
        
            # add extra distance cost
            if track_det_dists != [] and self.use_distance_prior:
                temp_ = torch.full_like(cost_mat[:num_trk,:num_det],0)
                # print(track_det_dists.shape,temp_.shape)
                # 99.99% of nusc cars travel slower than 22m/s
                temp_[torch.where(track_det_dists > 22)] = 3
                cost_mat[:num_trk,:num_det] += temp_


            if len(tracker.tracking_decisions) > 0 and len(tracker.detection_decisions) > 0:
                cost_mat[num_trk:,num_det:] = supervise['cost_mat'].clone().T
                # add extra distance cost
                if track_det_dists != [] and self.use_distance_prior:
                    temp_ = torch.full_like(cost_mat[:num_trk,:num_det],0)
                    # print(track_det_dists.shape,temp_.shape)
                    # 99.99% of nusc cars travel slower than 22m/s
                    temp_[torch.where(track_det_dists > 22)] = 3
                    cost_mat[num_trk:,num_det:] += temp_.T
                    

        if len(tracker.detection_decisions) > 0 and num_det > 0:
            det_costs, det_dec_idx = torch.min(
                torch.cat([supervise[k].unsqueeze(0) for k in tracker.detection_decisions],dim=0), dim=0
            )
            temp_idx = torch.arange(0,num_det,device=device)
            cost_mat[num_trk+temp_idx,temp_idx] = det_costs
        else:
            det_dec_idx = torch.tensor([],device=device)


        if len(tracker.tracking_decisions) > 0 and num_trk > 0:
            trk_costs, trk_dec_idx = torch.min(
                torch.cat([supervise[k].unsqueeze(0) for k in tracker.tracking_decisions],dim=0), dim=0
            )
            temp_idx = torch.arange(0,num_trk,device=device)
            cost_mat[temp_idx,num_det+temp_idx] = trk_costs
        else:
            trk_dec_idx = torch.tensor([],device=device)

        return (cost_mat, None, None), trk_dec_idx, det_dec_idx


    def get_cost_mat_softmax(self,num_det,num_trk,supervise,tracker,cost_mat,track_det_dists,device):
        decision_smx_cost = {}

        dd_mat = self.col_smx(
            torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(0) for k in tracker.detection_decisions ],dim=0)
        )
        decision_smx_cost.update({k:dd_mat[num_trk+i,:] for i,k in enumerate(tracker.detection_decisions)})

        td_mat = self.row_smx(
            torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(1) for k in tracker.tracking_decisions ],dim=1)
        )
        decision_smx_cost.update({k:td_mat[:,num_det+i] for i,k in enumerate(tracker.tracking_decisions)})

        if dd_mat.nelement() > 0:
            try:
                assert torch.max(dd_mat) <= 1.0, "torch.max(dd_mat) <= 1.0 == False"
                assert torch.min(dd_mat) >= 0.0, "torch.min(dd_mat) >= 0.0 == False"
            except AssertionError:
                print('max:',torch.max(dd_mat))
                print('Min:',torch.min(dd_mat))
                raise AssertionError

        if td_mat.nelement() > 0:
            try:
                assert torch.max(td_mat) <= 1.0, "torch.max(td_mat) <= 1.0 == False"
                assert torch.min(td_mat) >= 0.0, "torch.min(td_mat) >= 0.0 == False"
            except AssertionError:
                print('max:',torch.max(td_mat))
                print('Min:',torch.min(td_mat))
                raise AssertionError


        if num_det > 0 and num_trk > 0:
            # as in deep association networks
            cost_mat[:num_trk,:num_det] = torch.max(
                torch.cat([td_mat[:num_trk,:num_det].unsqueeze(0),dd_mat[:num_trk,:num_det].unsqueeze(0)],dim=0),dim=0
            ).values * -1.0 # for hungarian algorithm

            if len(tracker.tracking_decisions) > 0 and len(tracker.detection_decisions) > 0:
                cost_mat[num_trk:,num_det:] = cost_mat[:num_trk,:num_det].clone().T

        if len(tracker.detection_decisions) > 0 and num_det > 0:
            det_costs, det_dec_idx = torch.max(
                torch.cat([decision_smx_cost[k].unsqueeze(0) for k in tracker.detection_decisions],dim=0), dim=0
            )
            temp_idx = torch.arange(0,num_det,device=device)
            cost_mat[num_trk+temp_idx,temp_idx] = det_costs * -1.0
        else:
            det_dec_idx = torch.tensor([],device=device)
            

        if len(tracker.tracking_decisions) > 0 and num_trk > 0:
            trk_costs, trk_dec_idx = torch.max(
                torch.cat([decision_smx_cost[k].unsqueeze(0) for k in tracker.tracking_decisions],dim=0), dim=0
            )
            temp_idx = torch.arange(0,num_trk,device=device)
            cost_mat[temp_idx,num_det+temp_idx] = trk_costs * -1.0
        else:
            trk_dec_idx = torch.tensor([],device=device)

        return (cost_mat, dd_mat, td_mat), trk_dec_idx, det_dec_idx


    def get_cost_mat(self, *args, **kwargs):
        return getattr(self,"get_cost_mat_{}".format(self.cost_mat_type))(*args, **kwargs)
    



    def __call__(self, supervise, tracker, num_trk, num_det, class_name, track_det_dists, class_mask, device):
        """Method which applies hungarian matching to the detection and tracks

        Args:
            net (): network containing all differentiable components 
            class_name (): the names of the current class  
            trk_feats (torch.Tensor): the current track features
            det_feats (torch.Tensor): the current det features
            track_det_dists (torch.Tensor): distances betten the tracks and detections in the prev frame.
            device (torch.devics): the device to use

        Returns:
            trk_unmatched (list): the indices of active tracks which were not matched
            trk_new (list): the indices of current detections which were not matched
            det_match (list): the indices of current detections which were matched
            trk_match (list): the indices of active tracks which were matched
        """
        summary = {}
        # print({k:v.shape for k,v in supervise.items()})
        det_mul = 0 if len(tracker.detection_decisions) == 0 else 1
        trk_mul = 0 if len(tracker.tracking_decisions) == 0 else 1

        cost_mat = torch.full((num_trk + det_mul * num_det, num_det + trk_mul * num_trk,),
                              10000.0,
                              dtype=torch.float32,
                              device=device)

        with torch.no_grad():
            (cost_mat, dd_mat, td_mat) , trk_dec_idx, det_dec_idx = self.get_cost_mat(num_det=num_det,
                                                                    num_trk=num_trk,
                                                                    supervise=supervise,
                                                                    tracker=tracker,
                                                                    cost_mat=cost_mat,
                                                                    track_det_dists=track_det_dists,
                                                                    device=device)
            
                
                

        # print(cost_mat)
        numpy_cost_mat = cost_mat.detach().clone().cpu().numpy()
        numpy_cost_mat[num_trk:,num_det:] += class_mask.T.cpu().numpy()
        numpy_cost_mat[:num_trk,:num_det] += class_mask.cpu().numpy()

        trk_ind, det_ind = linear_sum_assignment(numpy_cost_mat)
        trk_ind = torch.from_numpy(trk_ind).to(device)
        det_ind = torch.from_numpy(det_ind).to(device)

        # get indices of actual decisions
        use_idx = torch.where(reduce(torch.logical_or,[trk_ind < num_trk , det_ind < num_det]))
        trk_ind = trk_ind[use_idx]
        det_ind = det_ind[use_idx]

        if len(tracker.detection_decisions) > 0 and len(tracker.tracking_decisions) > 0:
            assert cost_mat.size(0) == cost_mat.size(1)
        elif len(tracker.detection_decisions) > 0:
            assert len(det_ind) == num_det
        elif len(tracker.tracking_decisions) > 0:
            assert len(trk_ind) == num_trk

        # assert no incorrect decision
        incorrect_idx = torch.where(cost_mat[trk_ind,det_ind] == 10000.0)[0]
        assert len(incorrect_idx) == 0, "Error len(incorrect_idx) > 0"

       
        matches = torch.where(reduce(torch.logical_and,[trk_ind < num_trk, det_ind < num_det]))
        decisions = {}
        decisions['det_match'] = det_ind[matches]
        decisions['track_match'] = trk_ind[matches]

        detection_decisions_ = det_ind[
                torch.where(num_trk <= trk_ind)
            ]
        detection_decisions_idx = det_dec_idx[detection_decisions_]

        for i,k in enumerate(tracker.detection_decisions):
            decisions[k] = detection_decisions_[
                torch.where(detection_decisions_idx == i)
            ]

        tracking_decisions_ = trk_ind[
                torch.where(num_det <= det_ind)
            ]
        tracking_decisions_idx = trk_dec_idx[tracking_decisions_]
        
        for i,k in enumerate(tracker.tracking_decisions):
            decisions[k] = tracking_decisions_[
                torch.where(tracking_decisions_idx == i)
            ]
        
        if num_trk > sum( [ len(decisions['track_match']) ] + [ len(decisions[k]) for k in tracker.tracking_decisions ]):
            # print('in unmatched tracks')
            #unmatched tracks
            temp = torch.zeros(num_trk)
            temp[
                torch.cat([decisions['track_match']] + [decisions[k] for k in tracker.tracking_decisions])
            ] = 1
            decisions['track_unmatched'] = torch.where(temp == 0)[0]

        else:
            decisions['track_unmatched'] = torch.tensor([],dtype=torch.long,device=device)
        
        if num_det >  sum( [ len(decisions['det_match']) ] + [ len(decisions[k]) for k in tracker.detection_decisions ]):
            #unmatched detections
            temp = torch.zeros(num_det)
            temp[
                torch.cat([decisions['det_match']] + [decisions[k] for k in tracker.detection_decisions])
            ] = 1
            decisions['det_unmatched'] = torch.where(temp == 0)[0]
        else:
            decisions['det_unmatched'] = torch.tensor([],dtype=torch.long,device=device)


        assert decisions['det_unmatched'].nelement() == 0
        assert decisions['track_unmatched'].nelement() == 0

        summary.update(self.create_summary(tracker,supervise,class_name))

        try:
            assert num_det == sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) , "Det error"
            assert num_trk == sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']), 'track error'
        except AssertionError as e:
            print(e)
            print("[AssertionError] assert num_det == sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) ")
            print('or')
            print("[AssertionError] num_trk == sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track'])  ")
            print('      decisions:',[[k,x.size(0)] for k,x in decisions.items()],sum([x.size(0) for k,x in decisions.items()]))
            print('decisions det  : ',sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) )
            print('decisions track: ',sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']) )
            print('num_det',num_det)
            print('num_trk',num_trk)
            print('len(incorrect_idx)',len(incorrect_idx))
            print("len(trk_ind)",len(trk_ind))
            print("len(det_ind)",len(det_ind))

            print("sum( [ len(decisions['track_match']) ] + [ len(decisions[k]) for k in tracker.tracking_decisions ])",sum( [ len(decisions['track_match']) ] + [ len(decisions[k]) for k in tracker.tracking_decisions ]))
            exit(0)

            

        return decisions, summary, (numpy_cost_mat, dd_mat, td_mat)
        


    def create_summary(self,tracker,supervision,class_name):
        summary = {}
        for k in tracker.detection_decisions:
            if supervision[k].nelement() == 0:
                continue
            summary["{}_mean_cost_{}".format(class_name,k)] = torch.mean(supervision[k]).item()
            summary["{}_min_cost_{}".format(class_name,k)] = torch.min(supervision[k]).item()
            summary["{}_median_cost_{}".format(class_name,k)] = torch.median(supervision[k]).item()

        for k in tracker.tracking_decisions:
            if supervision[k].nelement() == 0:
                continue
            summary["{}_mean_cost_{}".format(class_name,k)] = torch.mean(supervision[k]).item()
            summary["{}_min_cost_{}".format(class_name,k)] = torch.min(supervision[k]).item()
            summary["{}_median_cost_{}".format(class_name,k)] = torch.median(supervision[k]).item()

        for k in ['match']:
            if supervision['cost_mat'].nelement() == 0:
                continue
            summary["{}_mean_cost_{}".format(class_name,k)] = torch.mean(supervision['cost_mat']).item()
            summary["{}_min_cost_{}".format(class_name,k)] = torch.min(supervision['cost_mat']).item()
            summary["{}_median_cost_{}".format(class_name,k)] = torch.median(supervision['cost_mat']).item()

        return summary