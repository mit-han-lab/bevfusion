import torch
from mmdet3d.models.builder import TRACKERS
from .track import Track




@TRACKERS.register_module()
class TrackingUpdater():
    """
    Class designed to update the track objects of the tracker 
    """

    def __init__(self,update_config):
        """
        decision_correct_sample_rate (dict): The rate of correct decisions for each decision type
        """
        super().__init__()
        self.update_config = update_config


    def __call__(self,tracker,pred_cls,det_confidence,trackCount,bbox,decisions,sample_token,refine_preds,forecast_preds,timestep,ego,device):
        """ Method for updating existing tracks

        Args:
            TODO

        Returns:
            trackCount (int): the new total number of tracks
        """
        track_update = {}
        active_track_update = []
        decom_track_update = []
        output_track_update = []
        
        for k in tracker.detection_decisions: #follow sorted order
            idx = decisions[k]
            track_update[k] = []
            if len(idx) != 0:
                track_update[k] += [len(tracker.tracks)+i for i in range(len(idx))]
                tracker.tracks += [Track(id_=trackCount + ii, 
                                    cls=pred_cls[idx[ii]], 
                                    sample=sample_token, 
                                    bbox=x.clone(),
                                    det_confidence=det_confidence[idx[ii]],
                                    score=refine_preds[idx[ii],-1].detach().clone(),
                                    futures=forecast_preds[idx[ii],:].detach().clone(),
                                    xy=refine_preds[idx[ii],-3:-1].detach().clone(),
                                    timestep=timestep,) 
                            for ii,x in enumerate(bbox[idx])]

                trackCount += len(idx)

            active_track_update, decom_track_update, output_track_update = self.update_tracks_decision(
                k, active_track_update, decom_track_update, output_track_update, track_idx=track_update[k]
            )


        tempTensorToTrack = {i:v for i,v in enumerate(tracker.activeTracks)}
        track_match_update = []
        #update existing tracks
        for deti,trki in zip(decisions['det_match'].cpu().numpy(),decisions['track_match'].cpu().numpy()):            
            idx = tempTensorToTrack[trki]
            tracker.tracks[idx].addTimestep(
                            sample=sample_token,
                            bbox=bbox.tensor[deti,:].clone(),
                            cls=pred_cls[deti],
                            det_confidence=det_confidence[deti],
                            score=refine_preds[deti,-1].detach().clone(),
                            futures=forecast_preds[deti,:].detach().clone(),
                            xy=refine_preds[deti,-3:-1].detach().clone(),
                            timestep=timestep,
                        )
            track_match_update.append(idx)

        active_track_update, decom_track_update, output_track_update = self.update_tracks_decision(
               'track_match', active_track_update, decom_track_update, output_track_update, track_idx=track_match_update
            )


        for k in tracker.tracking_decisions + ["track_unmatched"]:
            idx = decisions[k]
            track_update[k] = []
            active_track_update, decom_track_update, output_track_update = self.update_tracks_decision(
                k, active_track_update, decom_track_update, output_track_update, track_idx=decisions[k],
                tracker=tracker,ego=ego,device=device,sample_token=sample_token, timestep=timestep,
                tempTensorToTrack=tempTensorToTrack,
            )



        # print("In updater, active_track_update:",sorted(active_track_update))
        tracker.activeTracks = active_track_update
        # print("In updater, tracker.activeTracks:",sorted(tracker.activeTracks))
        tracker.decomTracks += decom_track_update
        tracker.non_max_suppression(device=device)
        
        return trackCount, output_track_update


    def update_tracks_decision(self,decision,*args,**kwargs):
        return getattr(self,'update_{}'.format(decision))(*args,**kwargs)


    def update_track_match(self, active_track_update, decom_track_update, output_track_update, track_idx, *args, **kwargs):
        config = self.update_config['match']

        return self.update_(config, track_idx, active_track_update, decom_track_update, output_track_update)


    def update_det_newborn(self, active_track_update, decom_track_update, output_track_update, track_idx, *args, **kwargs):
        config = self.update_config['det_newborn']

        return self.update_(config, track_idx, active_track_update, decom_track_update, output_track_update)

    def update_det_false_positive(self, active_track_update, decom_track_update, output_track_update, track_idx, *args, **kwargs):
        config = self.update_config['det_false_positive']

        return self.update_(config, track_idx, active_track_update, decom_track_update, output_track_update)


    def update_track_false_positive(self, active_track_update, decom_track_update, output_track_update, track_idx, tempTensorToTrack, *args, **kwargs):
        config = self.update_config['track_false_positive']

        track_idx = [tempTensorToTrack[i] for i in track_idx.cpu().numpy()]
        return self.update_(config, track_idx, active_track_update, decom_track_update, output_track_update)
        

    def update_track_false_negative(self, active_track_update, decom_track_update, output_track_update, 
                                    track_idx, tracker, device, ego, sample_token, timestep, tempTensorToTrack, *args, **kwargs):
        config = self.update_config['track_false_negative']

        for trki in track_idx.cpu().numpy():            
            idx = tempTensorToTrack[trki]
            tracker.tracks[idx].unmatched_step()
            if tracker.tracks[idx].unmatchedFrameCount >= tracker.frameLimit:
                if config['decom']:
                    decom_track_update.append(idx)
            else:
                
                tracker.tracks[idx].add_false_negative_timestep(
                        ego=ego,
                        sample=sample_token,
                        timestep=timestep,
                        device=device,
                        propagation_method=tracker.propagation_method,
                    )

                if config['output']:
                    output_track_update.append(idx)
                    
                if config['active']:
                    active_track_update.append(idx)

        # print("In update_track_false_negative",active_track_update )

        return active_track_update, decom_track_update, output_track_update



    def update_track_unmatched(self, active_track_update, decom_track_update, output_track_update, 
                                track_idx, tracker, tempTensorToTrack, ego, sample_token, timestep, device, *args, **kwargs):
        config = self.update_config['track_unmatched']

        for i in track_idx.cpu().numpy():
            idx = tempTensorToTrack[i]
            # print("\n[INFO] in update_track_unmatched for track id:",idx)
            
            tracker.tracks[idx].unmatched_step()
            if tracker.tracks[idx].unmatchedFrameCount >= tracker.frameLimit:
                # print('\n[INFO] in decom tracks for track id:',idx)
                decom_track_update.append(idx)
            else:
                # print('\n[INFO] in active and output tracks for track id:',idx)
                # tracker.tracks[idx].add_false_negative_timestep(
                #         ego=ego,
                #         sample=sample_token,
                #         timestep=timestep,
                #         device=device
                #     )
                # output_track_update.append(idx)
                active_track_update.append(idx)

        # print("In update_track_unmatched",active_track_update )

        return active_track_update, decom_track_update, output_track_update


    def update_(self,config, track_idx, active_track_update, decom_track_update, output_track_update ):
        if config['decom'] and config['active']:
            raise ValueError("Cannot be both active and decom")


        if config['output']:
            output_track_update += track_idx

        if config['decom']:
            decom_track_update += track_idx

        if config['active']:
            active_track_update += track_idx

        return active_track_update, decom_track_update, output_track_update
    


