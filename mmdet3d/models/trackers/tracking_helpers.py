import torch

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import reduce
from collections import namedtuple

TensorInfo = namedtuple("TensorInfo", "L1Norm ComponentMean Shape")



def getParamInfo(module,strng,grad=False):
    """Computes information about an nn.Module object's parameters

    Args:
        module (nn.Module): Module object of interest.
        strng (string): the string prefix used.
        grad (bool): if true computes the norm of the gradient, otherwise
            computes the norm of the parameters.

    Returns:
        d (dict): Map from inicative strings to their corresponding 
            TensorInfo object. 
    """
    d = {}
    for k,v in module._modules.items():
        if type(v) == torch.nn.Sequential:
            d.update(getParamInfo(v,strng+"_"+k,grad=grad))
        else:
            for key,t in v._parameters.items():
                kstr = "{}_{}_{}".format(strng,k,key)
                if grad:
                    kstr += '_grad'
                    tens = t.grad
                else:
                    tens = t
                tens = t.grad if grad else t
                if tens == None:
                    continue
                l1 = torch.norm(tens,p=1).item()
                shape = t.shape
                d[kstr] = TensorInfo(l1,l1/torch.prod(torch.tensor(shape)).item(), shape)
    return d


def update_log_vars(log_vars,log_vars_new):
    for k,v in log_vars_new.items():
        try:
            log_vars[k] += v
        except KeyError:
            log_vars[k] = v


def getTPS(det_bboxes,gt_bboxes,threshold,ioucal):
    """returns indices of the true positives"""
    iou = ioucal(det_bboxes,gt_bboxes)
    return torch.where(iou > threshold)



def torchInterp(x, source_range, target_range):
    return (x - source_range[0]) / (source_range[1] - source_range[0]) * (target_range[1] - target_range[0]) + target_range[0]


def interpolateBEV(bev_feats,xy,point_cloud_range,device):
    """returns indices of the true positives
    args:
        bev_feats (torch.tensor): tensor of size (C, H, W) containing features in BEV.
        xy (torch.tensor): tensor of size (N, 2) containing xy coordinates in the lidar frame.
        point_cloud_range (list[float]): dimensions of the point cloud input.
        device (torch.device): device on which the tensor is located.

    returns:
        intpolated (torch.tensor): tensor of size (N, C) containing the interpolated features.
    """
    
    assert len(bev_feats.shape) == 3
    assert xy.size(0) > 0 

    # normalize xy to bev dims
    xrange = [point_cloud_range[0],point_cloud_range[3]]
    yrange = [point_cloud_range[1],point_cloud_range[4]]

    x = torchInterp(xy[:,0], xrange, [0,bev_feats.shape[1]])
    y = torchInterp(xy[:,1], yrange, [0,bev_feats.shape[2]])
    grid = torch.cat([x.reshape(1,1,-1,1),y.reshape(1,1,-1,1)],dim=3)

    interpolated = torch.nn.functional.grid_sample(bev_feats.unsqueeze(0), 
                                                    grid, 
                                                    mode='bilinear', 
                                                    padding_mode='zeros', 
                                                    align_corners=False)
    return interpolated[0,:,0,:].permute(1,0)



def interpolate_bev_2d(bev_feats,xy,point_cloud_range,device):
    """returns indices of the true positives
    args:
        bev_feats (torch.tensor): tensor of size (C, H, W) containing features in BEV.
        xy (torch.tensor): tensor of size (N, 2) containing xy coordinates in the lidar frame.
        point_cloud_range (list[float]): dimensions of the point cloud input.
        device (torch.device): device on which the tensor is located.

    returns:
        intpolated (torch.tensor): tensor of size (N, C) containing the interpolated features.
    """
    
    assert len(bev_feats.shape) == 3
    assert xy.size(0) > 0 
    assert len(xy.shape) == 3
    b,n,_ = xy.shape

    # normalize xy to bev dims
    xrange = [point_cloud_range[0],point_cloud_range[3]]
    yrange = [point_cloud_range[1],point_cloud_range[4]]

    x = torchInterp(xy[...,0], xrange, [0,bev_feats.shape[1]])
    y = torchInterp(xy[...,1], yrange, [0,bev_feats.shape[2]])
    grid = torch.cat([x.reshape(1,b,n,1),y.reshape(1,b,n,1)],dim=3)
    
    interpolated = torch.nn.functional.grid_sample(bev_feats.unsqueeze(0), 
                                                   grid, 
                                                   mode='bilinear', 
                                                   align_corners=False)

    return interpolated[0,...].permute(1,2,0)




def get_bbox_sides_and_center(lidarbbox):
    corners = lidarbbox.corners
    #BL , BR , TL , TR
    x = corners[:,[0,4,7,3],:2]
    xc = corners[:,[4,7,3,0],:2]
    sides = (x + xc) / 2

    return torch.cat([lidarbbox.center[:,:2].unsqueeze(1),sides],dim=1)





def get_cost_mat_viz(tracker,tp_decisions,decisions,td_mat,dd_mat,cost_mat,num_track,num_det,device,
                     save_dir=None,show=False):
    cost_mat_ = torch.zeros((dd_mat.size(0),td_mat.size(1),),device=device)
    mask = torch.zeros((dd_mat.size(0),td_mat.size(1),),device=device)
    mask_gt = torch.zeros((dd_mat.size(0),td_mat.size(1),),device=device)
    
    cost_mat_[:num_track,:] = td_mat
    cost_mat_[:,:num_det] = dd_mat
    cost_mat_[:num_track,:num_det] = torch.from_numpy(cost_mat[:num_track,:num_det] * -1).to(device) # flip sign since this was used in hungarian matching
    cost_mat_[num_track:,num_det:] = 0
        
    mask[decisions['track_match'],decisions['det_match']] = 1
    mask_gt[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']] = 1
    
    for i,k in enumerate(tracker.detection_decisions):
        idx = decisions[k]
        mask[num_track+i,idx] = 1
        tp_idx = tp_decisions['pos_'+k]
        mask_gt[num_track+i,tp_idx] = 1
        
    for i,k in enumerate(tracker.tracking_decisions):
        idx = decisions[k]
        mask[idx,num_det+i] = 1
        tp_idx = tp_decisions['pos_'+k]
        mask_gt[tp_idx,num_det+i] = 1
    
    


    cost_mat_ = cost_mat_.cpu().numpy()
    mask = mask.cpu().numpy()
    mask_gt = mask_gt.cpu().numpy()

    ax = plt.subplot()
    im = ax.imshow(cost_mat_,
                   vmin=np.min(cost_mat_[cost_mat_ != -10000.]),
                   vmax=np.max(cost_mat_[cost_mat_ != -10000.]))
    
    label_map = {
        'track_false_positive':'Track FP',
        'track_false_negative':'Track FN',
        'det_newborn':'Det NB',
        'det_false_positive':'Det FP',
    }
    GT = np.where(mask == 1)
    ax.scatter(GT[1],GT[0],color='lime',marker="*",s=300)
    
    preds = np.where(mask == 1)
    ax.scatter(preds[1],preds[0],color='red')
    
    
    ax.set_yticklabels(['pad']+[f'Track{i}' for i in range(num_track)] + \
                       [label_map[x] for x in tracker.detection_decisions])
    ax.xaxis.tick_top() 
    ax.set_xticklabels(['pad']+[f'Det{i}' for i in range(num_det)] + \
                       [label_map[x] for x in tracker.tracking_decisions])
    
    ax.plot([-0.5, num_det - 0.5 + len(tracker.tracking_decisions)],
            [num_track - 0.5,num_track - 0.5],color='red')
    
    ax.plot([-0.5, num_det - 0.5],
            [num_track + 0.5,num_track + 0.5],color='red')
    
    ax.plot([num_det - 0.5, num_det - 0.5 ],
            [-0.5, num_track - 0.5 + len(tracker.detection_decisions) ],color='red')
    
    ax.plot([num_det + 0.5, num_det + 0.5 ],
            [-0.5, num_track - 0.5 ],color='red')
    
    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    if save_dir:
        plt.savefig(save_dir,bbox_inches='tight')
    
    if show:
        plt.show()
        


class EmptyBBox:
    def __init__(self,tensor):
        super().__init__()
        self.tensor = tensor



def log_mistakes(tracker,
                 tp_decisions,
                 decisions,
                 td_mat,
                 dd_mat,
                 cost_mat,
                 num_track,
                 num_det,
                 det_tp_idx,
                 active_tracks,
                 device):
    """
    Retrieve errors from predictions.
    """
    #matches: 100
    cost_mat_ = torch.zeros((dd_mat.size(0),td_mat.size(1),),device=device)
    mask = torch.zeros((dd_mat.size(0),td_mat.size(1),),device=device)
    mask_gt = torch.zeros((dd_mat.size(0),td_mat.size(1),),device=device)

    # print('cost_mat_ shape: ',cost_mat_.shape)
    # print('mask shape: ',mask.shape)
    # print('mask_gt shape: ',mask_gt.shape)
    
    cost_mat_[:num_track,:] = td_mat
    cost_mat_[:,:num_det] = dd_mat
    cost_mat_[:num_track,:num_det] = torch.from_numpy(cost_mat[:num_track,:num_det] * -1).to(device) # flip sign since this was used in hungarian matching
    cost_mat_[num_track:,num_det:] = 0
        
    mask[decisions['track_match'],decisions['det_match']] = 100
    mask_gt[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']] = 100
    
    dec_to_idx = {'match':100}
    for i,k in enumerate(tracker.detection_decisions):
        dec_to_idx[k] = len(dec_to_idx) + 1
        idx = decisions[k]
        mask[num_track+i,idx] = dec_to_idx[k]
        tp_idx = tp_decisions['pos_'+k]
        mask_gt[num_track+i,tp_idx] = dec_to_idx[k]
        
    for i,k in enumerate(tracker.tracking_decisions):
        dec_to_idx[k] = len(dec_to_idx) + 1
        idx = decisions[k]
        mask[idx,num_det+i] = dec_to_idx[k]
        tp_idx = tp_decisions['pos_'+k]
        mask_gt[tp_idx,num_det+i] = dec_to_idx[k]

    idx_to_dec = {v:k for k,v in dec_to_idx.items()}
    idx_to_dec[100] = 'match'

    # want distance to gt entry in the association matrix 
    det_lookup = torch.zeros(num_det,device=device)
    det_lookup[det_tp_idx] = 1

    track_lookup = torch.zeros(num_track,device=device)
    temp_id = tracker.trkid_to_gt_trkid[active_tracks]
    temp_tte = tracker.trkid_to_gt_tte[active_tracks]

    track_lookup[temp_id >= 0] = 1
    track_lookup[temp_tte < 0] = 2 # tte=0 is the last frame of the track
    track_lookup[temp_id == -1] = 0


    mask = mask.cpu()
    mask_gt = mask_gt.cpu()
    cost_mat_ = cost_mat_.cpu()

    for k in ['match'] + tracker.detection_decisions + tracker.tracking_decisions:
        # incorrect matches
        idx = torch.where(reduce(torch.logical_and,[mask_gt != dec_to_idx[k], mask == dec_to_idx[k]]))
        # print('\n\nidx',idx)
        # print(len(idx[0]))
        if len(idx[0]) == 0:
            continue

        for r,c in zip(idx[0],idx[1]):
            if c < num_det:
                correct_det_idx = torch.where(mask_gt[:,c] != 0)[0]
                det_mistake = idx_to_dec[mask_gt[correct_det_idx,c].item()]
                cost_gt = cost_mat_[correct_det_idx,c]
                cost_pred = cost_mat_[r,c]
                cost_max = torch.max(cost_mat_[:,c])
                # det_cost_diff = cost_mat_[r,c] - cost_mat_[correct_det_idx,c] # difference between the predicted cost and the cost of the correct entry
                # det_tp_diff_max = torch.max(cost_mat_[:,c]) - cost_mat_[correct_det_idx,c] # difference between the max cost and the cost of the correct entry
                det_type = 'TP' if det_lookup[c] == 1 else 'FP'

                print('k',k)
                print('det_mistake',det_mistake)
                try:
                    tracker.mistakes_det[k][det_mistake]['count'] += 1
                    tracker.mistakes_det[k][det_mistake]['cost_pred'].append(cost_pred)
                    tracker.mistakes_det[k][det_mistake]['cost_gt'].append(cost_gt)
                    tracker.mistakes_det[k][det_mistake]['cost_max'].append(cost_max)
                    tracker.mistakes_det[k][det_mistake]['det_type'][det_type] += 1
                    tracker.mistakes_det[k][det_mistake]['det_type_list'].append(det_type)
                except KeyError:
                    temp = {'FP':0, 'TP':0}
                    temp[det_type] = 1
                    tracker.mistakes_det[k][det_mistake] = {'count':1,
                                                            'cost_pred':[cost_pred],
                                                            'cost_gt':[cost_gt],
                                                            'cost_max':[cost_max],
                                                            'det_type_list':[det_type],
                                                            'det_type':temp,}
                    

            if r < num_track:
                correct_track_idx = torch.where(mask_gt[r,:] != 0)[0]
                # print('correct_track_idx',correct_track_idx)
                track_mistake = idx_to_dec[mask_gt[r,correct_track_idx].item()]
                # track_cost_diff = cost_mat_[r,c] - cost_mat_[r,correct_track_idx]
                # track_tp_diff_max = torch.max(cost_mat_[r,:]) - cost_mat_[r,correct_track_idx]
                cost_gt = cost_mat_[r,correct_track_idx]
                cost_pred = cost_mat_[r,c]
                cost_max = torch.max(cost_mat_[r,:])

                if track_lookup[r] == 1:
                    track_type = 'TP' 
                elif track_lookup[r] == 2:
                    track_type = 'TP-OOR' 
                elif track_lookup[r] == 0:
                    track_type = 'FP'
                else:
                    raise ValueError('track_lookup[r] = {}'.format(track_lookup[r]))

                print('k',k)
                print('track_mistake',track_mistake)

                try:
                    tracker.mistakes_track[k][track_mistake]['count'] += 1
                    tracker.mistakes_track[k][track_mistake]['cost_pred'].append(cost_pred)
                    tracker.mistakes_track[k][track_mistake]['cost_gt'].append(cost_gt)
                    tracker.mistakes_track[k][track_mistake]['cost_max'].append(cost_max)
                    tracker.mistakes_track[k][track_mistake]['track_type'][track_type] += 1
                    tracker.mistakes_track[k][track_mistake]['track_type_list'].append(track_type)
                except KeyError:
                    temp = {'FP':0,'TP-OOR':0,'TP':0}
                    temp[track_type] = 1
                    tracker.mistakes_track[k][track_mistake] = {'count':1,
                                                                'cost_pred':[cost_pred],
                                                                'cost_gt':[cost_gt],
                                                                'cost_max':[cost_max],
                                                                'track_type_list':[track_type],
                                                                'track_type':temp,}


            if k == 'match':
                try:
                    tracker.mistakes_match['IDS_{}_to_{}'.format(det_type,track_type)] += 1
                except KeyError:
                    tracker.mistakes_match['IDS_{}_to_{}'.format(det_type,track_type)] = 1
                

