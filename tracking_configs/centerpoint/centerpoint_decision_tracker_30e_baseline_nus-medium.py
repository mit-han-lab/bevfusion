
model_config_file = '../_base_/trackers/pnp_net_256_lstm_bev-interpolate_transformer-2-2-8_decisions.py'
schedule_config_file = '../_base_/schedules/cyclic_tracking_30e.py'
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/nus-mini-track-3d.py',
    schedule_config_file,
    model_config_file
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# For nuScenes we usually do 10-class detection
class_names = [
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

additional_neptune_tags = ['det dec baseline','nms']

future_count=6
past_count=4
model = dict(
    type='CenterPointTracker',
    class_names=class_names,
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
                        point_cloud_range=point_cloud_range,
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
                     point_cloud_range=point_cloud_range,
                     tracked_classes=tracked_classes,
                     tracked_range=tracked_range,
                     cls_to_idx=cls_to_idx,
                     det_nms_threshold=0.1,
                     det_nms_ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                     
                     tracker=dict(type='VirtualTracker',
                                  detection_decisions=['det_newborn','det_false_positive'],
                                  tracking_decisions=[],
                                  teacher_forcing=False,
                                  use_nms=True,
                                  ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                                  suppress_threshold=0.1,
                                  frameLimit=1, 
                                  tp_threshold=0.01,
                                  updater=dict(type='TrackingUpdater',
                                               update_config=dict(
                                                    det_newborn=dict(decom=False, output=True, active=True),
                                                    det_false_positive=dict(decom=False, output=False, active=False),
                                                    track_false_negative=dict(decom=False, output=True, active=True),
                                                    track_false_positive=dict(decom=True, output=False, active=False),
                                                    track_unmatched=dict(decom=True, output=False, active=False),
                                                    match=dict(decom=False, output=True, active=True),
                                                )),

                                  associator=dict(type='TrackingAssociator',use_distance_prior=False,cost_mat_type='margin'),
                                  incrementor=dict(type='TrackingIncrementorLSTM'),
                                  modifier=dict(type='TrackingDecisionModifier',
                                                decision_sampling=dict(det_newborn='linear_decay',
                                                                        track_false_positive='linear_decay',
                                                                        track_false_negative='linear_decay',
                                                                        det_false_positive='linear_decay',
                                                                        det_match='linear_decay'),),

                                  track_supervisor=dict(type='TrackingSupervisor',
                                                        association=dict(type='MarginAssociationSupervisor',
                                                                         balance_supervision=False,
                                                                         losses={'all':dict(type='MarginRankingLoss', margin=0.2, reduction='mean'),
                                                                                'det_match-det_newborn':dict(type='MarginRankingLoss', margin=0.2, reduction='mean'),
                                                                                'det_match-det_false_positive':dict(type='MarginRankingLoss', margin=0.2, reduction='mean'),
                                                                                'det_newborn-det_false_positive':dict(type='MarginRankingLoss', margin=0.2, reduction='mean'),
                                                                                },
                                                                         compute_summary=True),  
                                                        forecast=dict(type='ForecastingSupervisor',
                                                                    loss=dict(type='SmoothL1Loss'),
                                                                    compute_summary=True), 

                                                        refinement=dict(type='RefinementSupervisor',
                                                                        compute_summary=True,
                                                                        ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                                                                        regressionLoss=dict(type='SmoothL1Loss'),
                                                                        confidenceLoss=dict(type='MarginRankingLoss', margin=0.2, reduction='mean'))),),
))

work_dir='work_dirs'
dataset_type = 'NuScenesDataset_tracking'
data_root = 'data/nuscenes'
file_client_args = dict(backend='disk')
tracking_nuscenes_prefix = '/nuscenes_tracking/nuscenes-tracking-medium'
workflow = [('train', 1)]#,('val',1)]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D_tracking', 
         with_bbox_3d=True, 
         with_label_3d=True, 
         with_tracking=True,
         with_pasts=True,
         with_futures=True,
         with_track_tte=True),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[1., 1.],
        translation_std=[0, 0, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter_tracking', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter_tracking', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D_tracking', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_tracks' , 'gt_futures', 'gt_pasts', 'gt_track_tte'])
]


gpus=4
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='SequentialDataset',
        visualize=False,
        sample_size=16, #maxout for entire sequence
        sampling_mode='train',
        extra_sequence_multipliter=2,
        gpus=gpus,
        classes=class_names,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + tracking_nuscenes_prefix + '_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            filter_empty_gt=False,
            use_valid_flag=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),

    val=dict(
        type='SequentialDataset',
        visualize=False,
        # sample_size=16, #maxout for entire sequence (used when samplingmode is train)
        sampling_mode='val',
        gpus=gpus,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + tracking_nuscenes_prefix + '_infos_val.pkl',
            pipeline=train_pipeline,#test_pipeline,
            classes=class_names,
            test_mode=False,
            filter_empty_gt=False,
            use_valid_flag=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'))
        )


dataloader_shuffle=False
custom_hooks = [
    dict(type='CustomEval', priority='NORMAL', interval=10, eval_at_zero=False, eval_start_epoch=0),
    dict(type='ShuffleDatasetHook', priority='NORMAL'),
    dict(type='SaveModelToNeptuneHook', priority=40),
    dict(type='SetEpochInfoHook', priority=1),
]

neptune_source_files = ['configs'+x[2:] for x in _base_ ] 

log_config = dict(interval=16,
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type="NeptuneLoggerHook",
                        init_kwargs={    
                            'project':"bentherien/Tracking",
                            'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNWQ4NTRlNi00ZGJlLTRhZDctYmRlOC0zZWM4NmE3YWE0MWIifQ==",
                            'name': "[Schedule] {} [Model] {} ".format(schedule_config_file.split('/')[-1],
                                                                    model_config_file.split('/')[-1]),
                            'source_files': neptune_source_files,
                            'tags': ['v1.0-medium'] + additional_neptune_tags
                            },
                        interval=16,
                        ignore_last=True,
                        reset_flag=True,
                        with_step=True,
                        by_epoch=True)
            ])



checkpoint_config = dict(interval=100,save_last=True)
evaluation = dict(interval=100000, pipeline=[])
find_unused_parameters=True


# no_validate=True
# load_from = '/mnt/10g_bnas/neptune_checkpoints/saved_model (4).pth'
#load_from =  './work_dirs/epoch_500.pth'