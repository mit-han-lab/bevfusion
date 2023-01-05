import torch
import numpy as np
import torch.nn as nn
from .transforms import affine_transform, apply_transform
from pyquaternion import Quaternion

class Track(nn.Module):
    """Represents one individual track."""

    def __init__(self, id_, cls, sample, bbox, score, futures, xy, timestep, det_confidence):
        """Creates a Track object.

        BBOX Format: [xs, ys, hei, dim, rot, vel]
        xs: 1
        ys: 1
        hei: 1
        dim: 3
        rot: 1
        vel: 2
        total: 9

        Args:
            id_ (int): the current track's global id
            cls (string): the current tracks's class type
            sample (string): Token id of the current sample 
            bbox (torch.Tensor): the current track's predicted bounding box
        """
        super().__init__()
        # print("In Track init, timestep: ", timestep)

        if timestep == -1:
            raise ValueError("Timestep cannot be -1")

        self.id = id_
        self.cls = [cls]
        self.samples = [sample]
        self.refined_bboxes = [bbox]
        self.det_bboxes = [bbox.clone()]
        self.det_confidence = [det_confidence * 0.01]
        self.scores = [score]
        self.futures = [futures.reshape(-1,2)]
        self.timestep_to_idx = {timestep:0}
        self.refine(xy)
        self.deactivated = False
        self.unmatchedFrameCount = 0

        assert score > 0.0, "Score must be greater than 0.0 but was {}".format(score)

    def __str__(self):
        return "Track(id:{},cls:{},samples: {}, bboxes:{})".format(
            self.id, self.cls, type(self.samples), type(self.refined_bboxes)
        )
    
    def __del__(self):
        del self.id
        del self.cls
        del self.samples
        del self.det_confidence
        del self.refined_bboxes
        del self.det_bboxes
        del self.deactivated
        del self.unmatchedFrameCount
        del self.futures
        del self.timestep_to_idx

    
        
    def addTimestep(self,cls,sample,bbox,det_confidence,score,futures,xy,timestep):
        """Adds a timestep to the tracks lists.

         Args:
            sample (string): Token id of the current sample 
            bbox (torch.Tensor): the current track's predicted bounding box
        """
        if len(self.scores) == 1:
            self.det_confidence[0] = self.det_confidence[0] / 0.01

        #reset frame limit 
        # self.frame_limit = 0

        self.timestep_to_idx[timestep] = len(self.scores)
        self.cls.append(cls)
        self.samples.append(sample)
        self.refined_bboxes.append(bbox)
        self.det_bboxes.append(bbox.clone())
        self.det_confidence.append(det_confidence)
        self.scores.append(score)
        try:
            self.futures.append(futures.reshape(-1,2))
        except RuntimeError:
            print(futures)
            raise RuntimeError
            exit(0)

        self.refine(xy)

    def add_false_negative_timestep(self,sample,timestep,ego,device,propagation_method='velocity'):
        """Adds a timestep with an incremented track bbox based on the vehicle's
        predicted trajectory.
        """
        prev = list(self.timestep_to_idx.keys())[-1]

        self.timestep_to_idx[timestep] = len(self.scores)
        self.samples.append(sample)

        refined_temp, det_temp = self.transform_over_time(from_=prev,to_=timestep,ego=ego,propagation_method=propagation_method)

        self.refined_bboxes.append(torch.tensor(refined_temp,dtype=torch.float32,device=device))
        self.det_bboxes.append(torch.tensor(det_temp,dtype=torch.float32,device=device))
        self.scores.append(0.01 * self.scores[-1])
        self.det_confidence.append(0.01 * self.det_confidence[-1]) #same update as in simple track 
        self.futures.append(None)
        self.cls.append(self.cls[-1])


    def transform_over_time(self,from_,to_,ego,propagation_method='velocity'):
        """Transforms the track's bounding box over time and increments it.

        Args:
            to_ (int): the timestep to transform to
            from_ (int): the timestep to transform from
            ego (Ego): the ego vehicle
        """
        diff = to_ - from_
        idx = self.timestep_to_idx[from_]
        det_temp = self.det_bboxes[idx].clone().cpu().numpy()
        refined_temp = self.refined_bboxes[idx].clone().cpu().numpy()
        
        if propagation_method == 'future':
            try:
                #obtain regressed future for the timestep
                #TODO replace this with a fucnction for doing this...
                future = self.futures[idx][diff-1,:].cpu().numpy()
                future = np.concatenate([future,[0]])[np.newaxis,:]
                # print('diff',diff)
                # print('future',future)
                aff_r = affine_transform(
                    rotation=np.roll(Quaternion(axis=[0, 0, 1], angle=((refined_temp[6] + np.pi/2 ) * -1)).elements,-1),
                    rotation_format='quat',
                    translation=refined_temp[:3],
                )
                xyz_r = apply_transform(aff_r,#np.linalg.inv(aff_r),
                                        future)
                # print('xyz_r',xyz_r)
                
                aff_d = affine_transform(
                    rotation=np.roll(Quaternion(axis=[0, 0, 1], angle=det_temp[6]).elements,-1),
                    rotation_format='quat',
                    translation=det_temp[:3],
                )
                xyz_d = apply_transform(aff_d, #np.linalg.inv(aff_d),
                                        future)
                
                # print("using future prediction")
                
            except IndexError:
                #happens when no future has the correct previous timestep
                future = diff * ( self.det_bboxes[idx][-2:].cpu().numpy() / 2 ) #div by 2 for m/0.5s
                future = np.concatenate([future,[0]])[np.newaxis,:]
                xyz_d = det_temp[:3] + future
                xyz_r = refined_temp[:3] + future
            except TypeError:
                #happens when prev step was a False negative an future is none
                future = diff * ( self.det_bboxes[idx][-2:].cpu().numpy() / 2 ) #div by 2 for m/0.5s
                future = np.concatenate([future,[0]])[np.newaxis,:]
                xyz_d = det_temp[:3] + future
                xyz_r = refined_temp[:3] + future
        elif propagation_method == 'velocity':
            future = diff * ( self.det_bboxes[idx][-2:].cpu().numpy() / 2 ) #div by 2 for m/0.5s
            future = np.concatenate([future,[0]])[np.newaxis,:]
            xyz_d = det_temp[:3] + future
            xyz_r = refined_temp[:3] + future
        else:
            raise ValueError('propagation_method must be either "future" or "velocity", got {}'.format(propagation_method))


        xyz = np.concatenate([xyz_d[np.newaxis,:],xyz_r[np.newaxis,:]],axis=0)
        xyz = ego.transform_over_time(xyz,from_=from_,to_=to_)

        refined_temp[:3] = xyz[0,:]
        det_temp[:3] = xyz[1,:]

        return refined_temp, det_temp

    def __getitem__(self, timestep):
        # print("inside __getitem__(self, timestep)")
        try:
            bbox_out = self.det_bboxes[self.timestep_to_idx[timestep]]
        except KeyError:
            print('KEY ERROR in GETitem')
            print('timestep',timestep)
            print('unmatchedframecount',self.unmatchedFrameCount)
            print('timestep_to_idx',self.timestep_to_idx)
            print('id',self.id)
            raise KeyError

        return bbox_out
    
    def refine(self, xy):
        """Refines the track's xy position

        Args:
            xy (torch.Tensor): the current track's refined xy offset prediction
        """
        self.refined_bboxes[-1][:2] = self.refined_bboxes[-1][:2] + xy
    

    def deactivate(self):
        self.deactivated = True
        
    def unmatched_step(self):
        self.unmatchedFrameCount += 1
    
        