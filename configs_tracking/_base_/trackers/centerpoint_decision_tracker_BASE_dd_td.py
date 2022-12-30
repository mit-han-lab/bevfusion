# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range_arg = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# For nuScenes we usually do 10-class detection
class_names_arg = [
    'car', 'truck', 
    'construction_vehicle', 'bus', 
    'trailer', 'barrier',
    'motorcycle', 'bicycle', 
    'pedestrian', 'traffic_cone'
]

tracked_range = {'car':50, 'truck':50, 'bus':50, 'trailer':50,
                 'motorcycle':40, 'bicycle':40, 'pedestrian':40}

cls_to_idx = {
    'car':0, 'truck':1, 
    'construction_vehicle':2, 'bus':3, 
    'trailer':4, 'barrier':5, 
    'motorcycle':6, 'bicycle':7, 
    'pedestrian':8, 'traffic_cone':9
}

tracked_classes = [
    'car', 'truck', 
    'bus', 'trailer',
    'motorcycle', 'bicycle', 
    'pedestrian'
]

tracked_classes = ['car']


model = dict(
    type='CenterPointTracker',
    class_names=class_names_arg,
    compute_loss_det=False,
    use_neck=dict(use=False),
    use_backbone=dict(use=False),
    use_middle_encoder=dict(use=False),

    load_detector_from='/btherien/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth', 
    pretrained_config='configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_tracking_detector.py', 
    test_output_config={'bbox':'track', 'score':'det'},
    bev_supervisor=dict(type='BEVSupervisor',
                        class_indices=[cls_to_idx[class_name] for class_name in tracked_classes],
                        loss=dict(type='TripletMarginLoss', margin=0.5, reduction='mean'),
                        forecast_supervisor=dict(type='ForecastingSupervisor',
                                                  loss=dict(type='SmoothL1Loss'),
                                                  compute_summary=True),
                        use_metric=True,
                        use_forecast=True,
                        point_cloud_range=point_cloud_range_arg,
                        compute_summary=True,
                        sample_num=2,
                        subsample=False
                        ),
    net = dict(type='DecisionTracker',
                merge_forward='interpolate',
                message_passing_forward='simple',
                decisions_forward={'det_newborn':'MLP',
                                    'det_false_positive':'MLP',
                                    'track_false_negative':'MLP',
                                    'track_false_positive':'MLP',
                                    'match':'MLP'},),

    trk_manager=dict(type='TrackingManager',
                     point_cloud_range=point_cloud_range_arg,
                     tracked_classes=tracked_classes,
                     tracked_range=tracked_range,
                     cls_to_idx=cls_to_idx,
                     det_nms_threshold=0.1,
                     det_nms_ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                     use_det_nms=True,
                     
                     tracker=dict(type='VirtualTracker',
                                  detection_decisions=['det_newborn','det_false_positive'],
                                  tracking_decisions=['track_false_negative','track_false_positive'],
                                  teacher_forcing=False,
                                  use_nms=True,
                                  ioucal=dict(type='Center2DRange', distance=2.0),
                                  #ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                                  suppress_threshold=0.1,#track NMS
                                  frameLimit=1, 
                                  tp_threshold=0.01,
                                  updater=dict(type='TrackingUpdater',
                                               update_config=dict(
                                                    det_newborn=dict(decom=False, output=True, active=True),
                                                    det_false_positive=dict(decom=True, output=False, active=False),
                                                    track_false_negative=dict(decom=False, output=True, active=True),
                                                    track_false_positive=dict(decom=True, output=False, active=False),
                                                    track_unmatched=dict(decom=True, output=False, active=False),
                                                    match=dict(decom=False, output=True, active=True),
                                                )),

                                  associator=dict(type='TrackingAssociator', use_distance_prior=False),
                                  incrementor=dict(type='TrackingIncrementorLSTM'),
                                  modifier=dict(type='TrackingDecisionModifier',
                                                decision_sampling=dict(det_newborn='linear_decay',
                                                                        track_false_positive='linear_decay',
                                                                        track_false_negative='linear_decay',
                                                                        det_false_positive='linear_decay',
                                                                        det_match='linear_decay'),),

                                  track_supervisor=dict(type='TrackingSupervisor',
                                                        association=dict(),  
                                                        forecast=dict(type='ForecastingSupervisor',
                                                                    loss=dict(type='SmoothL1Loss'),
                                                                    compute_summary=True), 

                                                        refinement=dict(type='RefinementSupervisor',
                                                                        compute_summary=True,
                                                                        ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                                                                        regressionLoss=dict(type='SmoothL1Loss'),
                                                                        confidenceLoss=dict(type='MarginRankingLoss', margin=0.2, reduction='mean'))),),
))
