import numpy as np
import torch

import torch.nn as nn
from mmdet3d.models.builder import TRACKERS

from functools import reduce
from scipy.optimize import linear_sum_assignment


@TRACKERS.register_module()
class TrackingIncrementorLSTM():

    def __init__(self):
        super().__init__()
        self.prev_c = None
        self.prev_h = None
        self.track_feats = None

    def reset(self,net,device):
        try:
            del self.prev_c
            del self.prev_h
            del self.track_feats
        except AttributeError:
            pass

        self.prev_c = torch.empty((0,net.lstm.hidden_size),dtype=torch.float32,device=device)
        self.prev_h = torch.empty((0,net.lstm.hidden_size),dtype=torch.float32,device=device)
        self.track_feats = torch.empty((0,net.lstm.hidden_size),dtype=torch.float32,device=device)


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


    def gatherAny(self,feats,index):
        """Indexes into the appropriate tensor to gather the desired
        freatures."""
        out = []
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


    def __call__(self,net,active_tracks,hidden_idx,increment_hidden_idx,new_idx,feats,device):
        """Updates the saved LSTM state of matched tracks and computes
        the first step for new tracks.

        Args:
            net (torch.tensor): nn.Module object containing all the networks
            hidden_idx (torch.tensor): Matched track indices, corresponding to det_match.
            increment_hidden_idx (torch.tensor) correspoind to hidden_idx, these are the inputs to tracks 
                which should be incremented.
            new_idx (torch.tensor): New track indices, corresponding to features in 'increment_feats'.
            feats (torch.tensor):
            device (torch.device): input features to the LSTM

        Returns: Nothing
        """
        new_num = len(new_idx)        # number of new tracks to add
        increment_num = len(increment_hidden_idx)   # number of existing tracks to increment

        try:
            assert increment_num == len(hidden_idx)
        except AssertionError:
            print('increment_num != len(hidden_idx)')
            print(increment_num,len(hidden_idx))
            print(increment_hidden_idx,hidden_idx)
            exit(0)

        if new_num == 0 and increment_num == 0:
            return 

        if increment_num == 0:
            h_new = net.h0.weight.unsqueeze(0).repeat(1,new_num,1)
            c_new = net.c0.weight.unsqueeze(0).repeat(1,new_num,1)
            
            feats_new = feats[new_idx].unsqueeze(0) #get the features of new dets

            out_trk, (out_h, out_c) = net.lstm(feats_new,(h_new,c_new))
            
            assert out_trk.size(1) == new_num

            if active_tracks == [] and self.track_feats == None:
                try:
                    assert self.track_feats == None
                    assert self.prev_h == None
                    assert self.prev_c == None
                except AssertionError:
                    print('active_tracks == [] but self.track_feats != None')
                    print(self.track_feats.shape,self.prev_h.shape,self.prev_c.shape)
                    exit(0)

                self.track_feats = out_trk.squeeze(0).clone()
                self.prev_h = out_h.squeeze(0).clone()
                self.prev_c = out_c.squeeze(0).clone()
            else:
                self.track_feats = torch.cat([self.track_feats,out_trk[0,:,:]],dim=0).clone()
                self.prev_h = torch.cat([self.prev_h,out_h[0,:,:]],dim=0).clone()
                self.prev_c = torch.cat([self.prev_c,out_c[0,:,:]],dim=0).clone()
        else:
            h_trk = self.gatherAny(['h'],active_tracks)[hidden_idx].unsqueeze(0)
            c_trk = self.gatherAny(['c'],active_tracks)[hidden_idx].unsqueeze(0)
            feats_in = feats[increment_hidden_idx].unsqueeze(0) #get the features of matched dets
            
            if new_num > 0:
                h_new = net.h0.weight.unsqueeze(0).repeat(1,new_num,1)
                c_new = net.c0.weight.unsqueeze(0).repeat(1,new_num,1)
                feats_new = feats[new_idx].unsqueeze(0) #get the features of new dets

                h_trk = torch.cat([h_trk,h_new],dim=1)
                c_trk = torch.cat([c_trk,c_new],dim=1)
                feats_in = torch.cat([feats_in,feats_new],dim=1)

            out_trk, (out_h, out_c) = net.lstm(feats_in,(h_trk,c_trk))

            assert out_trk.size(1) == increment_num + new_num

            #concatenated the new tracks to the existing track tensors
            if new_num > 0:
                self.track_feats = torch.cat([self.track_feats,out_trk[0,-new_num:,:]],dim=0).clone()
                self.prev_h = torch.cat([self.prev_h,out_h[0,-new_num:,:]],dim=0).clone()
                self.prev_c = torch.cat([self.prev_c,out_c[0,-new_num:,:]],dim=0).clone()
            
            prev_trk_update_idx = torch.tensor(active_tracks,dtype=torch.long,device=device)[hidden_idx]
            assert len(prev_trk_update_idx) == increment_num
            
            #replace previous tensors in memory with new tracks
            self.track_feats[prev_trk_update_idx] = out_trk[:,:increment_num,:].clone()
            self.prev_h[prev_trk_update_idx] = out_h[:,:increment_num,:].clone()
            self.prev_c[prev_trk_update_idx] = out_c[:,:increment_num,:].clone()