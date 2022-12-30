import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.core.bbox.iou_calculators import BboxOverlapsNearest3D, BboxOverlaps3D

import torch
from torch.nn import TripletMarginLoss
from torch.nn.utils.rnn import pad_sequence
from mmdet3d.models.trackers.tracking_helpers import interpolateBEV
from mmdet3d.models.builder import SUPERVISORS, build_supervisor

from mmdet.core.bbox.iou_calculators.builder import build_iou_calculator

class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor, weight=None):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=weight,
            reduction = self.reduction

        )


LOSSES = {
    'TripletMarginLoss': TripletMarginLoss,
    'SmoothL1Loss':nn.SmoothL1Loss,
    'CrossEntropyLoss':nn.CrossEntropyLoss,
    'MarginRankingLoss':nn.MarginRankingLoss,
    'FocalLoss':FocalLoss,
    'nll_loss':nn.NLLLoss,
    'BCELoss':nn.BCELoss,
    'BCEWithLogitsLoss':nn.BCEWithLogitsLoss,
}



def build_loss(dict):
    if dict == None:
        return None

    type = dict.pop('type')
    try:
        return LOSSES[type](**dict)
    except KeyError:
        raise KeyError('Unrecognized loss type {}'.format(type))



@SUPERVISORS.register_module()       
class CpointSupervisor:
    def __init__(self,
                 forecast_supervisor,
                 bce_loss,
                 ioucal,
                 use_forecast=True,
                 use_confidence=True,
                 compute_summary=True):
        log, summary = dict(), dict()
        if forecast_supervisor:
            self.forecast_supervisor = build_supervisor(forecast_supervisor)
        else:
            self.forecast_supervisor = None
            
        if ioucal:
            self.ioucal = build_iou_calculator(ioucal)
        else:
            self.ioucal = None
            
        self.bce_loss = build_loss(bce_loss)
        
        
    def __call__(self,net,bev_feats,pred_bboxes,gt_bboxes,gt_futures,device):
        if bev_feats.size(0) == 0 or gt_bboxes.tensor.size(0) == 0:
            return torch.tensor(0.,dtype=torch.float32,device=device), dict(), dict()

        iou = self.ioucal(gt_bboxes.tensor,pred_bboxes.tensor)
        pred_iou,pred_idx = torch.max(iou,dim=1)
        gt_idx = torch.arange(0,gt_bboxes.tensor.size(0))

        target = torch.min(torch.tensor([1.],device=device),
                            torch.max(torch.tensor([0.],device=device),
                                        torch.tensor([2.],device=device) * pred_iou - torch.tensor([0.5],device=device))
                            ).reshape(-1)


        confidence_pred = net.bev_confidence_proj(bev_feats[pred_idx,:]).reshape(-1)
        
        assert confidence_pred.shape == target.shape
        bce_loss_ = self.bce_loss(confidence_pred,target)


        loss = torch.tensor(0.,dtype=torch.float32,device=device)
        loss = loss + bce_loss_
        
        if self.forecast_supervisor:
            forecast_loss, summary, log = self.forecast_supervisor(tp_det_idx=pred_idx,
                                                                   tp_gt_idx=gt_idx,
                                                                   forecast_preds=net.MLPPredict(net.bev_forecast_proj(bev_feats)),
                                                                   gt_futures=gt_futures,
                                                                   device=device,
                                                                   log_prefix='transformer_',
                                                                   return_loss=True)
        
            loss = loss + forecast_loss
            
        return loss, summary, log




@SUPERVISORS.register_module()
class BEVSupervisor:
    
    def __init__(self,
                 class_indices,
                 forecast_supervisor=None,
                 loss=dict(type='TripletMarginLoss'),
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 use_metric=True,
                 use_forecast=True,
                 compute_summary=True,
                 sample_num=2,
                 subsample=True,):
        """Supervises the BEV features of the tracker.
        Args:
            class_indices (list): list of class indices to supervise
            loss (str): loss to use for metric learning 
            point_cloud_range (list): range of the point cloud
            forecast (bool): whether to use the forecasted features
            metric (bool): whether to use triplet loss
            compute_summary (bool): whether to compute summary statistics
        Returns:
            None
        """
        
        self.point_cloud_range = point_cloud_range
        self.compute_summary = compute_summary
        self.class_indices = class_indices
        self.use_metric = use_metric
        self.use_forecast = use_forecast
        self.sample_num = sample_num
        self.subsample = subsample
        self.memory = {}

        if forecast_supervisor:
            self.forecast_supervisor = build_supervisor(forecast_supervisor)
        else:
            self.forecast_supervisor = None

        self.loss = build_loss(loss)
            
    def reset(self):
        print("Resetting memory for BEVSupervisor")
        del self.memory
        self.memory = {}
        for i in self.class_indices:
            self.memory[f'{i}feats_mem'] = []
            self.memory[f'{i}trk_id_mem'] = []
            
    
    def get_metric_loss(self,device):
        def subsample_negatives(current_idx,uniques,device,sample_num=2):
            offset = torch.tensor([x.size(0) for x in uniques],dtype=torch.long,device=device)
            indices = torch.tensor([],dtype=torch.long,device=device)
            for i,x in enumerate(uniques):
                if i == current_idx:
                    pass
                else:
                    actual_sample_num = min(sample_num,x.size(0))
                    samples = torch.multinomial(torch.ones(x.size(0),device=device), actual_sample_num, replacement=True)
                    offs = offset[:i].sum() if i <= current_idx else offset[:i].sum() - offset[current_idx]
                    indices = torch.cat([indices,
                                         samples + offs])
            return indices
    
    
        loss = torch.tensor(0.,requires_grad=True,device=device)
        for i in self.class_indices:
            # print([x.unsqueeze(0).shape for x in self.memory[f'{i}trk_id_mem']])
            # print([x.unsqueeze(0) for x in self.memory[f'{i}trk_id_mem']])
            trk_idx = pad_sequence([x.unsqueeze(1) for x in self.memory[f'{i}trk_id_mem']],
                                   batch_first=True,
                                   padding_value=-1.0).squeeze(2)
            trk_feats = pad_sequence([x.unsqueeze(1) for x in self.memory[f'{i}feats_mem']],
                                     batch_first=True,
                                     padding_value=-1.0).squeeze(2)

            
            u = torch.unique(trk_idx.flatten())
            uniques = []
            
            #get different tracks
            for x in u.cpu().numpy():
                if x == -1:
                    pass
                else:
                    idx = torch.where(trk_idx == x)
                    temp = trk_feats[idx[0],idx[1],...]
                    assert len(temp.shape) == 2
                    uniques.append(temp) #note order is not needed here
        
            #iter over tracks 
            for i,tens in enumerate(uniques):
                feat_dim = tens.shape[-1]
                num_tracks = tens.size(0)
                temp_idx = torch.arange(0,num_tracks,device=device)
                negatives = torch.cat([x for idx,x in enumerate(uniques) if idx != i],dim=0)
                
                a, p, n = [], [], []
                for anchor in range(num_tracks):
                    pos_ = tens[torch.where(temp_idx != anchor)[0],...]
                    
                    if self.subsample:
                        # print('subsampling...')
                        neg_idx_to_use = subsample_negatives(i,uniques,device,sample_num=self.sample_num)
                        cp = torch.cartesian_prod(torch.arange(0,pos_.size(0),device=device),neg_idx_to_use)
                        # print(cp.shape)
                    else:
                        cp = torch.cartesian_prod(torch.arange(0,pos_.size(0),device=device),
                                                  torch.arange(0,negatives.size(0),device=device))
                        
                    # print("Repeating anchors for each positive and negative: ",cp.size(0))
                    a.append(tens[anchor,:].repeat(cp.size(0),1))
                    p.append(pos_[cp[:,0],:])
                    n.append(negatives[cp[:,1],:])

                a = torch.cat(a,dim=0)
                p = torch.cat(p,dim=0)
                n = torch.cat(n,dim=0)  

                # dont compute loss for empty tensors
                if a.size(0) == 0 or p.size(0) == 0 or n.size(0) == 0:
                    continue

                metric_loss = self.loss(anchor=a,
                                        positive=p,
                                        negative=n)

                
                if metric_loss.isnan().any():
                    print("NAN in metric loss")
                    print('a',torch.cat(a).shape)
                    print('p',torch.cat(p).shape)
                    print('n',torch.cat(n).shape)
                    print("Loss",metric_loss)
                    # exit(0)

                loss = loss + metric_loss
        return loss
                
    
    def __call__(self,forecast_net,pts_feats,gt_labels,gt_bboxes,gt_tracks,
                 gt_futures,first_in_scene,last_in_scene,device,log_prefix=''):
        """Supervises the BEV features of the tracker."""

        if not self.use_forecast and not self.use_metric:
            return [torch.tensor(0.,requires_grad=True,device=device) for x in range(pts_feats.size(0))], {}

        summary, log = {}, {}
        losses = []
        batch = len(pts_feats)
        for i in range(batch):
            if first_in_scene[i] and self.use_metric:
                self.reset()

            forecast_loss = torch.tensor(0.,requires_grad=True,device=device)
            for cls_idx in self.class_indices:
                bbox_idx = torch.where(gt_labels[i] == cls_idx)
                if bbox_idx[0].size(0) == 0:
                    continue
                
                bev_feats = interpolateBEV(bev_feats=pts_feats[i,...],
                                           xy=gt_bboxes[i][bbox_idx][:,:2] + ( torch.randn(bbox_idx[0].size(0),2,device=device)/2 ),
                                           point_cloud_range=self.point_cloud_range,
                                           device=device)
                
                if self.use_metric:
                    self.memory[f'{i}feats_mem'].append(bev_feats)
                    self.memory[f'{i}trk_id_mem'].append(gt_tracks[i][bbox_idx])
                    
                if self.use_forecast:
                    
                    assert bev_feats.size(0) == len(bbox_idx[0])
                    if len(bbox_idx[0]) > 0:
                        forecast_loss, summary_forecast, log_forecast = self.forecast_supervisor(
                                            tp_det_idx=torch.arange(0,bev_feats.size(0)),
                                            tp_gt_idx=bbox_idx[0],
                                            forecast_preds=forecast_net(bev_feats),
                                            gt_futures=gt_futures[i],
                                            device=device,
                                            log_prefix=log_prefix)
                        summary.update(summary_forecast)
                        log.update(log_forecast)
                    
                # if self.compute_summary:
                #     summary[f'{log_prefix}forecast_loss_cls_{i}'] = forecast_loss.item()
                    
            try:
                losses[i] = losses[i] + forecast_loss
            except IndexError: 
                losses.append(forecast_loss)

        
        if last_in_scene[i] and self.use_metric:
            metric_loss = self.get_metric_loss(device)
            if self.compute_summary:
                summary[f'{log_prefix}metric_loss'] = metric_loss.item()
            losses[i] = losses[i] + metric_loss
            
        return losses, summary, log



from mmcv.utils.config import ConfigDict


@SUPERVISORS.register_module()
class TrackingSupervisor:
    """Class for supervissing different tracking losses"""
    def __init__(self, association=None, 
                       forecast=None, 
                       refinement=None):




        self.association = self.build_module(association)
        self.forecast = self.build_module(forecast)
        self.refinement = self.build_module(refinement)
        # exit(0)

        # if association:
        #     self.association = build_supervisor(association)
        # else:
        #     self.association = None

        # if forecast:
        #     self.forecast = build_supervisor(forecast)
        # else:
        #     self.forecast = None

        # if refinement:
        #     self.refinement = build_supervisor(refinement)
        # else:
        #     self.refinement = None

    def build_module(self, cfg):
        print(type(cfg),cfg)
        if type(cfg) == list:
            return [build_supervisor(x) for x in cfg]
        elif type(cfg) == dict or type(cfg) == ConfigDict:
            return build_supervisor(cfg)
        else:
            print("TrackingSupervisor: No module found.\n\n\n\n")
            print("################################################################")
            return None

    def exec_module(self, module, *args, **kwargs):
        loss = 0
        summary, log = {}, {}
        if type(module) == list:
            for x in module: 
                temp_loss, temp_summary, temp_log = x(*args, **kwargs)
                summary.update(temp_summary)
                log.update(temp_log)
                loss = loss + temp_loss
            return loss, summary, log
        elif module == None:
            return None
        else:
            return module(*args, **kwargs)

    def supervise_association(self, *args, **kwargs):
        return self.exec_module(self.association, *args, **kwargs)
        # return self.association(*args, **kwargs)

    def supervise_forecast(self, *args, **kwargs):
        return self.exec_module(self.forecast, *args, **kwargs)
        # return self.forecast(*args, **kwargs)

    def supervise_refinement(self, *args, **kwargs):
        # print(self.refinement)
        return self.exec_module(self.refinement, *args, **kwargs)
        # return self.refinement(*args, **kwargs)




@SUPERVISORS.register_module()
class MarginAssociationSupervisor:
    def __init__(self, losses,lambda_pn=1.0,lambda_pos=1.0, compute_summary=True, 
                 balance_supervision=False, use_orig_loss=False, use_pos=False,
                 weights=dict(match=1.0,det_false_positive=1.0,track_false_positive=1.0,
                              track_false_negative=1.0,det_newborn=1.0,)):
        """Supervision module for association loss.
        
        args:
            loss (str): loss function to use.
            margins: (dict): margins for margin loss.
            compute_summary (bool): whether to compute summary.
        """
        self.compute_summary = compute_summary
        self.loss = {k:build_loss(loss_) for k,loss_ in losses.items()}
        self.balance_supervision = balance_supervision
        self.use_orig_loss = use_orig_loss
        self.use_pos = use_pos
        self.weights = weights
        self.lambda_pn = lambda_pn
        self.lambda_pos = lambda_pos

        for k in losses.keys():
            try:
                self.weights[k]
            except KeyError:
                self.weights[k] = 1.0

        if self.use_orig_loss == True and self.use_pos == False:
            raise ValueError("Pos loss must be enabled if using orig loss")

    def get_positive_to_negative_sup(self,tracker,tp_decisions,supervise,device,return_loss=True):
        """Retrieves column and row wise supervision for margin loss."""
        num_trk, num_det = supervise['cost_mat'].shape

        dd_mat = torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(0) for k in tracker.detection_decisions ] ,dim=0)
        td_mat = torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(1) for k in tracker.tracking_decisions ]  ,dim=1)


        dd_mask = torch.zeros_like(dd_mat)
        dd_mask[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']] = 1
        for i,k in enumerate(tracker.detection_decisions):
            dd_mask[num_trk + i, tp_decisions['pos_'+k]] = 1


        td_mask = torch.zeros_like(td_mat)
        td_mask[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']] = 1
        for i,k in enumerate(tracker.tracking_decisions):
            td_mask[tp_decisions['pos_'+k], num_det + i] = 1
    
        cp_outer = torch.tensor([],dtype=torch.float32,device=device)

        cp_outer = {k:torch.tensor([],dtype=torch.float32,device=device) for k in ['det_match'] + tracker.tracking_decisions + tracker.detection_decisions}

        if num_trk > 0 and num_det > 0:

            for r,c in zip(tp_decisions['pos_track_match'],tp_decisions['pos_det_match']):
                pos = dd_mat[r,c].reshape(1)
                dd_neg = dd_mat[torch.where(dd_mask[:,c] == 0)[0],c]
                td_neg = td_mat[r,torch.where(td_mask[r,:] == 0)[0]]
                # print(pos.shape,torch.cat([dd_neg,td_neg]).shape)

                cp = torch.cartesian_prod(pos,torch.cat([dd_neg,td_neg]))
                
                if cp_outer['det_match'].nelement() == 0:
                    cp_outer['det_match'] = cp
                else:
                    cp_outer['det_match'] = torch.cat([cp_outer['det_match'],cp],dim=0)

        if num_det > 0:

            for i,k in enumerate(tracker.detection_decisions):
                for idx in tp_decisions['pos_'+k]:
                    # print(num_trk + i,idx)
                    pos = dd_mat[num_trk + i,idx].reshape(1)
                    dd_neg = dd_mat[torch.where(dd_mask[:,idx] == 0)[0],idx]
                    cp = torch.cartesian_prod(pos,dd_neg)

                    if cp_outer[k].nelement() == 0:
                        cp_outer[k] = cp
                    else:
                        cp_outer[k] = torch.cat([cp_outer[k],cp],dim=0)



        if num_trk > 0:

            for i,k in enumerate(tracker.tracking_decisions):
                for idx in tp_decisions['pos_'+k]:
                    # print(idx, num_det + i)
                    pos = td_mat[idx, num_det + i].reshape(1)
                    td_neg = td_mat[idx,torch.where(td_mask[idx,:] == 0)[0]]
                    cp = torch.cartesian_prod(pos,td_neg)

                    if cp_outer[k].nelement() == 0:
                        cp_outer[k] = cp
                    else:
                        cp_outer[k] = torch.cat([cp_outer[k],cp],dim=0)

        return cp_outer



    def __call__(self,tracker,tp_decisions,supervise,device,return_loss=True):
        """Method for supervising the association obtained from 
            hungarian matching. 

               D E T S
              # # # # #
            t # # # # #
            r # # # # #
            k # # # # #
            s # # # # #
        
        Args:
            tracker (torch.tensor): indices of the TP detections from predictions
            tp_decisions (torch.tensor): indices of the GT detections 
            supervise (torch.tensor): previsous active tracks
            return_loss (int): number of tracks
            device (torch.device): the device to run the code on

        Returns:
            margin_loss (torch.Tensor): negative loss for the association
            margin_loss_pos (torch.Tensor): positive loss for the association
            summary (dict): dictionary containing loggable statistics
            update_ (torch.Tensor): updated track idx to gt track idx mapping
        """
        summary, log = {}, {}
        margin_loss = torch.tensor(0.,requires_grad=True,device=device)

        if not return_loss:
            return margin_loss, summary, log

        cost_mat = supervise['cost_mat']
        pos_sup_targets = dict(det_match=cost_mat[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']])
        neg_sup_targets = dict(det_match=cost_mat[tp_decisions['neg_track_match'],tp_decisions['neg_det_match']])

        if self.use_orig_loss:

            if self.balance_supervision:
                max_num = pos_sup_targets['det_match'].size(0)
                for k in tracker.tracking_decisions + tracker.detection_decisions:
                    if max_num == 0:
                        pos_sup_targets[k] = torch.tensor([],device=device)
                        continue
                    pos_sup_targets[k] = supervise[k][tp_decisions['pos_'+k]]
                    size = pos_sup_targets[k].size(0)
                    if size > 0:
                        pos_sup_targets[k] = pos_sup_targets[k][torch.multinomial(torch.ones(size,device=device),min(max_num,size))]

            else:
                pos_sup_targets.update({k:supervise[k][tp_decisions['pos_'+k]] for k in tracker.tracking_decisions + tracker.detection_decisions})

            neg_sup_targets.update({k:supervise[k][tp_decisions['neg_'+k]] for k in tracker.tracking_decisions + tracker.detection_decisions})

            all_pos_sup = torch.cat([v for v in pos_sup_targets.values()],dim=0)
            all_neg_sup = torch.cat([v for v in neg_sup_targets.values()],dim=0)
            
            margin_loss = self.compute_cp_and_margin_loss(margin_key='all', 
                                                          pos=all_pos_sup, 
                                                          neg=all_neg_sup, 
                                                          device=device)

        else:
            cp_outer = self.get_positive_to_negative_sup(tracker,tp_decisions,supervise,device,return_loss)
            for k,v in cp_outer.items():
                if v.nelement() == 0:
                    continue

                temp_loss  = self.loss['pn_'+k](v[:,0],
                                                v[:,1],
                                                torch.full_like(v[:,1],-1.,device=device))

                margin_loss = margin_loss + temp_loss * self.weights['pn_'+k]


            pos_sup_targets.update({k:supervise[k][tp_decisions['pos_'+k]] for k in tracker.tracking_decisions + tracker.detection_decisions})
            #neg_sup_targets.update({k:supervise[k][tp_decisions['neg_'+k]] for k in tracker.tracking_decisions + tracker.detection_decisions})

        pos_margin_loss = torch.tensor(0.,requires_grad=True,device=device)
        if self.use_pos:
            #seperate distances between decisions
            for k in self.loss.keys():
                if k == 'all':
                    continue
                elif k.startswith('pn_'):
                    continue

                temp_loss = self.compute_cp_and_margin_loss(margin_key=k, 
                                                            pos=pos_sup_targets[k.split('-')[0]], 
                                                            neg=pos_sup_targets[k.split('-')[1]], 
                                                            device=device)
                if self.compute_summary:
                    summary["margin_loss_"+k] = temp_loss.item()

                pos_margin_loss = pos_margin_loss + temp_loss * self.weights[k]


        margin_loss = margin_loss * self.lambda_pn \
                      + pos_margin_loss * self.lambda_pos

        return margin_loss, summary, log
    

    def get_acc(self,gt,preds):
        if len(gt) == 0:
            return -1, 0
        _, first_idx, second_idx = np.intersect1d(gt,preds,return_indices=True)
        return len(first_idx) / len(gt), len(first_idx)


    def compute_cp_and_margin_loss(self, margin_key, pos, neg, device):
        if pos.size(0) > 0 and neg.size(0) > 0:
            cp = torch.cartesian_prod(pos,neg)
            margin_loss = self.loss[margin_key](cp[:,0],
                                                cp[:,1],
                                                torch.full_like(cp[:,0],-1.,device=device))

            
            margin_loss = margin_loss / len(pos)

        else:
            margin_loss = torch.tensor(0.,device=device, requires_grad=True)

        return margin_loss


    def show_summary(self,summary):
        for k,v in summary.items():
            print(k,v) 



@SUPERVISORS.register_module()
class FocalLossAssociationSupervisor:
    def __init__(self, focal_loss, l1_loss, losses=None, compute_summary=True, balance_supervision=False, 
                 weights=None,loss_weights=None):
        """Supervision module for association loss.
        
        args:
            loss (str): loss function to use.
            margins: (dict): margins for margin loss.
            compute_summary (bool): whether to compute summary.
        """
        self.focal_loss = build_loss(focal_loss)
        self.l1_loss = build_loss(l1_loss)
        self.compute_summary = compute_summary
        self.weights = weights
        self.loss_weights = loss_weights

    def __call__(self,tracker,tp_decisions,supervise,device,return_loss=True):
        """Method for supervising the association obtained from 
            hungarian matching. 

               D E T S
              # # # # #
            t # # # # #
            r # # # # #
            k # # # # #
            s # # # # #
        
        Args:
            tracker (torch.tensor): indices of the TP detections from predictions
            tp_decisions (torch.tensor): indices of the GT detections
            supervise (torch.tensor): ground truth tracks
            return_loss (int): whether to return a loss or not
            device (torch.device): the device to run the code on

        Returns:
            loss (torch.Tensor): loss
            summary (dict): dictionary containing loggable statistics
            log (torch.Tensor): dictionary containing loggable statistics
        """
        summary, log = {}, {}

        if not return_loss:
            margin_loss = torch.tensor(0.,requires_grad=True,device=device)
            return margin_loss, summary, log


        dd_mat = torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(0) for k in tracker.detection_decisions ] ,dim=0)
        td_mat = torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(1) for k in tracker.tracking_decisions ]  ,dim=1)

        

        num_trk, num_det = td_mat.size(0),dd_mat.size(1)


        if num_det > 0:
            if self.weights is not None:
                det_weights = torch.full((num_trk,),self.weights['match'],dtype=torch.float32,device=device)

            dd_gt = torch.zeros(num_det,dtype=torch.long,device=device)
            dd_gt[tp_decisions['pos_det_match']] = tp_decisions['pos_track_match']
            for i,k in enumerate(tracker.detection_decisions):
                dd_gt[tp_decisions['pos_'+k]] = num_trk + i
                if self.weights is not None:
                    det_weights = torch.cat([det_weights, torch.tensor([self.weights[k]],dtype=torch.float32,device=device)])

            focal_loss_det = self.focal_loss(dd_mat.T,dd_gt, weight= self.weights if self.weights is None else det_weights)
        else:
            focal_loss_det = torch.tensor(0.,requires_grad=True,device=device)

        if num_trk > 0:
            if self.weights is not None:
                track_weights = torch.full((num_det,),self.weights['match'],dtype=torch.float32,device=device)

            td_gt = torch.zeros(num_trk,dtype=torch.long,device=device)
            td_gt[tp_decisions['pos_track_match']] = tp_decisions['pos_det_match']
            for i,k in enumerate(tracker.tracking_decisions):
                td_gt[tp_decisions['pos_'+k]] = num_det + i
                if self.weights is not None:
                    track_weights = torch.cat([track_weights, torch.tensor([self.weights[k]],dtype=torch.float32,device=device)])

            focal_loss_trk = self.focal_loss(td_mat,td_gt, weight= self.weights if self.weights is None else track_weights)
        else:
            focal_loss_trk = torch.tensor(0.,requires_grad=True,device=device)

        

        trk_smx = F.log_softmax(
            torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(0) for k in tracker.detection_decisions ],dim=0)
        ,dim=0)
        

        det_smx = F.log_softmax(
            torch.cat([ supervise['cost_mat'] ] + [ supervise[k].unsqueeze(1) for k in tracker.tracking_decisions ],dim=1)
        ,dim=1)
        
        
        preds = (torch.exp(trk_smx[:num_trk,:num_det]) - torch.exp(det_smx[:num_trk,:num_det])).abs().flatten()
        consistency_loss = self.l1_loss(preds,torch.zeros_like(preds,device=device))

        if num_det > 0 and num_trk > 0:
            assemble_loss = -1.0 * torch.max(
                torch.cat([trk_smx[:num_trk,:num_det].unsqueeze(0),det_smx[:num_trk,:num_det].unsqueeze(0)],dim=0),dim=0
            ).values

            assemble_loss = assemble_loss.mean()
        else:
            assemble_loss = torch.tensor(0.,device=device, requires_grad=True)

        if self.loss_weights is not None:
            loss = focal_loss_det * self.loss_weights['det'] \
                   + focal_loss_trk * self.loss_weights['track'] \
                   + consistency_loss * self.loss_weights['consistency'] \
                   + assemble_loss * self.loss_weights['assemble']
        else:
            loss = focal_loss_det + focal_loss_trk + consistency_loss + assemble_loss

        if self.compute_summary:
            log["assemble_loss"] = assemble_loss.clone().detach().reshape(1)
            log["consistency_loss"] = consistency_loss.clone().detach().reshape(1)
            log["focal_loss_det"] = focal_loss_det.clone().detach().reshape(1)
            log["focal_loss_trk"] = focal_loss_trk.clone().detach().reshape(1)
            


        return loss, summary, log
    





@SUPERVISORS.register_module()
class ForecastingSupervisor:
    def __init__(self,loss=dict(type='SmoothL1Loss'),compute_summary=True):
        """Forecasting supervisor for the forecasting loss"""
        self.compute_summary = compute_summary
        self.loss = build_loss(loss)

    def __call__(self,tp_det_idx,tp_gt_idx,forecast_preds,gt_futures,device,log_prefix='',return_loss=True):
        """Method for supervising the forecasting process

        args:
            tp_det_idx: indices of true positive detections
            tp_gt_idx: indices of true positive ground truth tracks
            forecast_preds: predicted future tracks
            gt_futures: ground truth future tracks
            device: device to run on

        returns:
            forecast_loss: forecasting loss
            summary: dictionary of summary statistics
         
        """
        log, summary = {}, {}
        if return_loss == False:
            if len(tp_det_idx) > 0 and gt_futures != torch.Size([]):
                forecast_preds = forecast_preds.reshape(forecast_preds.size(0),-1,2) 

                assert forecast_preds[tp_det_idx].shape == gt_futures[tp_gt_idx].shape

                #get futures with supervision targets
                targets = gt_futures[tp_gt_idx,...]
                targets = targets.reshape(-1,2)
                mask = torch.where(targets != -5000.0)
                if targets[mask].shape != torch.Size([0]):
                    preds = forecast_preds[tp_det_idx].reshape(-1,2)
                    preds = preds[mask[0],:]
                    targets = targets[mask[0],:]
                    forecast_loss = self.loss(preds.flatten(),targets.flatten())
                    log[f"{log_prefix}ADE_forecast"] = torch.norm(preds - targets,p=2, dim=1)

            return torch.tensor(0.,device=device, requires_grad=True), summary, log

        if len(forecast_preds) == 0:
            return torch.tensor(0., device=device, requires_grad=True), summary, log


        if len(tp_det_idx) > 0 and gt_futures != torch.Size([]):
            forecast_preds = forecast_preds.reshape(forecast_preds.size(0),-1,2) 

            assert forecast_preds[tp_det_idx].shape == gt_futures[tp_gt_idx].shape

            #get futures with supervision targets
            targets = gt_futures[tp_gt_idx,...]
            targets = targets.reshape(-1,2)
            mask = torch.where(targets != -5000.0)

            # print(targets)
            

            if targets[mask].shape == torch.Size([0]):
                # Don't supervise if there are no future points
                forecast_loss = torch.tensor(0.,device=device, requires_grad=True)
            else:
                preds = forecast_preds[tp_det_idx].reshape(-1,2)
                preds = preds[mask[0],:]
                targets = targets[mask[0],:]
                forecast_loss = self.loss(preds.flatten(),targets.flatten())
                # print('mask',[x.shape for x in mask])
                # print('preds',preds.shape)
                # exit(0)

                if self.compute_summary:
                    with torch.no_grad():
                        log[f"{log_prefix}ADE_forecast"] = torch.norm(preds - targets,p=2, dim=1)
                        summary[f"{log_prefix}forecast_loss"] = forecast_loss.item()
                        # summary[f"{log_prefix}ADE_forecast"] = torch.mean(torch.norm(preds - targets,p=2, dim=1)).item()

            if str(forecast_loss.detach().clone().cpu().item()) == 'nan':
                print("############################")
                print("[ERROR]forecast_loss == nan")
                print("############################")
            elif len(tp_det_idx) > 0 and forecast_loss.item() == 0 and mask[0].shape != torch.Size([0]):
                print("[ERROR]len(tp_det_idx) > 0 and forecast_loss.item() == 0")
                print("mask",mask)
                print("gt_futures",gt_futures)
                print("gt_futures",forecast_preds)

        else:
            forecast_loss = torch.tensor(0.,device=device, requires_grad=True)
        
        return forecast_loss, summary, log


@SUPERVISORS.register_module()
class RefinementSupervisor:

    def __init__(self, regressionLoss, confidenceLoss, compute_summary=True, ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar')):
        self.compute_summary = compute_summary
        self.regressionLoss = build_loss(regressionLoss)
        self.confidenceLoss = build_loss(confidenceLoss)
        self.ioucal = build_iou_calculator(ioucal)


    def __call__(self, tp_det_idx, tp_gt_idx, refine_preds, refined_bboxes, bbox, gt_pasts, gt_bboxes, device, return_loss=True):
        """Method for supervising the refinement process"""
        summary = {}
        log = {}
        if return_loss == False:
            if len(tp_det_idx) > 0 and gt_pasts != torch.Size([]):
                refine_past = refine_preds[:,:-1].reshape(refine_preds.size(0),-1,2)

                #swap current refined xy to be offset xy
                gt_pasts[tp_gt_idx,-1,:] = gt_pasts[tp_gt_idx,-1,:] - bbox.tensor[tp_det_idx,:2]


                assert refine_past[tp_det_idx].shape == gt_pasts[tp_gt_idx].shape

                targets = gt_pasts[tp_gt_idx,...]
                targets = targets.reshape(-1,2)
                mask = torch.where(targets != -5000.0)

                # mask = torch.where(gt_pasts[tp_gt_idx] != -5000.0)
                preds = refine_past[tp_det_idx].reshape(-1,2)
                preds = preds[mask[0],:]
                targets = targets[mask[0],:]
                with torch.no_grad():
                    temp = torch.norm(preds - targets, p=2, dim=1)
                    log["ADE_refine"] = temp

            return torch.tensor(0., device=device, requires_grad=True), summary, log


        if len(bbox.tensor) == 0:
            return torch.tensor(0., device=device, requires_grad=True), summary, log





        if len(tp_det_idx) > 0 and gt_pasts != torch.Size([]):
            #remove scores 
            refine_past = refine_preds[:,:-1].reshape(refine_preds.size(0),-1,2)

            #swap current refined xy to be offset xy
            gt_pasts[tp_gt_idx,-1,:] = gt_pasts[tp_gt_idx,-1,:] - bbox.tensor[tp_det_idx,:2]


            assert refine_past[tp_det_idx].shape == gt_pasts[tp_gt_idx].shape

            targets = gt_pasts[tp_gt_idx,...]
            targets = targets.reshape(-1,2)
            mask = torch.where(targets != -5000.0)

            # mask = torch.where(gt_pasts[tp_gt_idx] != -5000.0)
            preds = refine_past[tp_det_idx].reshape(-1,2)
            preds = preds[mask[0],:]
            targets = targets[mask[0],:]
            #regress past
            refine_loss = self.regressionLoss(preds.flatten(),targets.flatten())

            if self.compute_summary:
                with torch.no_grad():
                    temp = torch.norm(preds - targets, p=2, dim=1)
                    log["ADE_refine"] = temp
                    # summary[f"ADE_refine"] = torch.mean(temp).item()
            
            refine_score = refine_preds[:,-1] #(x,y,score)
            
            try:
                assert len(refine_score) == len(refined_bboxes)
            except AssertionError:
                print(refine_score.shape,refined_bboxes.shape)
                exit(0)

            refined_iou = self.ioucal(refined_bboxes,gt_bboxes.tensor)

            #TODO change this max operation to a where? 
            if type(self.ioucal) == BboxOverlapsNearest3D:
                #when using iou
                m = torch.max(refined_iou, dim=1) # get highest iou entries per prediction
                sorted_ = torch.sort(m.values,descending=True) # sort by highest iou
                temp = sorted_.indices #prediction indices sorted by highest iou
            else:
                #when using euclidean distance
                m = torch.min(refined_iou, dim=1) # get smallest 2-norm entries per prediction
                sorted_ = torch.sort(m.values,descending=False) # sort by smallest 2-Norm distance
                temp = sorted_.indices #prediction indices sorted by highest iou

            cprod = None
            if temp.size(0) > 1:
                for i in range(temp.size(0) - 1):
                    inner_temp = temp[i+1:].unsqueeze(1)
                    repeat_temp = torch.full_like(inner_temp,temp[i],device=device)
                    if i == 0:
                        cprod = torch.cat([repeat_temp,inner_temp], dim=1)
                    else:
                        cprod = torch.cat([cprod,torch.cat([repeat_temp,inner_temp], dim=1)],dim=0)
          
                if cprod == None:
                    score_loss = torch.tensor(0., device=device, requires_grad=True)
                else:
                    try:
                        score_loss = self.confidenceLoss(refine_score[cprod[:,0]],
                                                         refine_score[cprod[:,1]],
                                                         torch.full_like(cprod[:,0],-1.,device=device))
                    except RuntimeError:
                        print("RuntimeError inside supervise_refinement() with RANK:",self.rank)
                        exit(0)

            else:
                score_loss = torch.tensor(0.,device=device, requires_grad=True)

            if self.compute_summary:
                summary['refine_loss'] = refine_loss.item()
                summary['confidence_score_loss'] = score_loss.item()

            refine_loss = refine_loss + score_loss

        else:
            refine_loss = torch.tensor(0.,device=device, requires_grad=True)


        if len(tp_det_idx) > 0 and refine_loss.item() == 0.:
            print("len(tp_det_idx) > 0 and forecast_loss.item() == 0")
            exit(0)

        return refine_loss, summary, log