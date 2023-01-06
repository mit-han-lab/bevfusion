import json 
import pickle
import os
import os.path as osp

import numpy as np
import torch
from scipy.interpolate import interpn, BPoly
import pytorch3d.transforms as transforms3d
import pytorch3d

from mmdet3d.ops import points_in_boxes_batch
from pytorch3d.structures.pointclouds import Pointclouds



def apply_rotation_to_angle(aff,euler):
    assert len(euler.shape) == 2, 'angle must have a batch dimension'
    aff_q = transforms3d.matrix_to_quaternion(aff[:,:3,:3])
    axis_angle = torch.zeros((euler.size(0),3),device=euler.device)
    axis_angle[:,2] = euler[:,0]
    angle_q = transforms3d.axis_angle_to_quaternion(axis_angle)
    res = transforms3d.quaternion_multiply(aff_q,angle_q)
    return transforms3d.quaternion_to_axis_angle(res)

def get_affine_torch(rotation,translation,device,angle_type='axis_angle'):
    assert len(rotation.shape) == 2, f'rotation must have batch dimension, got {rotation.shape}'
    assert len(translation.shape) == 2, f'translation must have batch dimension, got {translation.shape}'
    b = rotation.size(0)
    out = torch.zeros((b,4,4,), dtype=torch.float32, device=device)
    if angle_type == 'axis_angle':
        rot = transforms3d.axis_angle_to_matrix(rotation)
    elif angle_type == 'quaternion':
        rot = transforms3d.quaternion_to_matrix(rotation)
    else:
        raise NotImplemented(f'not implemented for angle_type: {angle_type}')
        
    out[:,:3,:3] = rot
    out[:,[0,1,2],3] = translation
    out[:,3,3] = 1.0 # homogeneous coords
    return out



def linear_interp_sweeps(sweeps_ts,xy1,xy2,ts1,ts2,device='cpu'):
    """Linearly interpolate between two xy points according to timestamps."""
    assert len(xy1.shape) == 2, f"xy1 must have a batch dimension, got size {xy1.shape}"
    assert len(xy2.shape) == 2, f"xy2 must have a batch dimension, got size {xy2.shape}"
    
    diff = ts2 - ts1
    ts_diff = ( sweeps_ts - ts1 ) / diff 
    xydiff = xy2[:,:2] - xy1[:,:2]
    return xy1[None,:,:2] + ( xydiff.reshape(1,-1,2) * ts_diff.reshape(-1,1,1))



def interpolate_per_frame(ts1,ts2,bboxes,offset,sweeps_infos,pts_sweeps,device):
    """Interpolates bounding boxes over sweeps and ls"""
    assert len(sweeps_infos) == 10
    assert bboxes.dtype == torch.float32
    assert pts_sweeps[0].dtype == torch.float32
    s_num = len(sweeps_infos)
    b_num = bboxes.size(0)
    b = s_num * b_num
    
    #Get interpolated bounding boxes
    sweeps_ts = torch.tensor([x['timestamp']/1e6 for x in sweeps_infos],device=device)
    interp_sweeps = linear_interp_sweeps(sweeps_ts=sweeps_ts,
                                         xy1=bboxes[:,:2] - offset,
                                         xy2=bboxes[:,:2],
                                         ts1=ts1/1e6,
                                         ts2=ts2/1e6,
                                         device=device)
    interp_bboxes = torch.cat([interp_sweeps,bboxes[None,:,2:7].repeat(10,1,1)],dim=2)
    
    assert interp_sweeps.shape == torch.Size((sweeps_ts.size(0),bboxes.size(0),2))
    assert interp_bboxes.shape == torch.Size((sweeps_ts.size(0),bboxes.size(0),7))

    #get points in boxes
    pc_batched = Pointclouds([x[:,:3].to(device) for x in pts_sweeps])
    pc_padded = pc_batched.points_padded()
    pib_idx = points_in_boxes_batch(pc_padded, interp_bboxes,).bool()
    
    points_in_boxes,lengths = [],[]
    for i in range(bboxes.size(0)):
        pib_temp = [pc_padded[s,pib_idx[s,:,i],:] for s in range(len(pts_sweeps))]
        points_in_boxes.extend(pib_temp)
        lengths.extend([x.size(0) for x in pib_temp])
    #create large batch
    points_batch = Pointclouds(points_in_boxes).points_padded()
    
    #create affine matrices to center points
    reshaped_bboxes = interp_bboxes.reshape(-1,7)
    rot = torch.zeros((b,3),dtype=torch.float32,device=device)
    rot[:,2] =  (reshaped_bboxes[:,6] + np.pi/2 ) * -1.
    interp_affines = get_affine_torch(translation=reshaped_bboxes[:,:3],
                                      rotation=rot,
                                      device=device,
                                      angle_type='axis_angle')
    
    #pad points to homogeneous coordinates
    points_batch = torch.cat(
        [points_batch,torch.ones(points_batch.shape[:2]+(1,),device=device)],dim=2
    )
    centered = torch.bmm(interp_affines,points_batch.permute(0,2,1)).permute(0,2,1)[:,:,:3]
    return centered.reshape(s_num,b_num,-1,3), torch.tensor(lengths,device=device).reshape(s_num,b_num)
    

def get_input_batch(centered,lengths,subsample_number,device):
    new_size = list(centered.shape)
    new_size[2] = subsample_number
    
    temp = [torch.randint(high=x, size=(1,512,))  if x != 0 else torch.zeros((1,512)) \
            for x in lengths.reshape(-1)]
    temp = torch.cat(temp).reshape(new_size[:2]+[-1]).long()
    
    where_pts = torch.where(lengths > 0)
    out = torch.zeros(new_size,dtype=torch.float32,device=device)
    out[where_pts[0],where_pts[1],:,:] = centered[
                                                      where_pts[0],
                                                      where_pts[1],
                                                      temp[where_pts[0],where_pts[1],:].t(),
                                                      :
                                                  ].permute(1,0,2)
    return out






