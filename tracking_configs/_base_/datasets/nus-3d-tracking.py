
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

dataset_type = 'NuScenesDataset_tracking'
data_root = 'data/nuscenes'
file_client_args = dict(backend='disk')
tracking_nuscenes_prefix = '/nuscenes_tracking/nuscenes-tracking'


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
