
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

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
tracking_nuscenes_prefix = '/nuscenes-tracking/bevfusion-tracking-medium'



reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
image_size = [256, 704]

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented,),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,),
    dict(type='LoadAnnotations3D_tracking', 
         with_bbox_3d=True, 
         with_label_3d=True, 
         with_tracking=True,
         with_attr_label=False,
         with_pasts=True,
         with_futures=True,
         with_track_tte=True),
    dict(type='ImageAug3D',
         final_dim=image_size,
         resize_lim=[0.48, 0.48],
         bot_pct_lim=[0.0, 0.0],
         rot_lim=[0.0, 0.0],
         rand_flip=False,
         is_train=False,),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim= [1.0, 1.0],
        rot_lim= [0.0, 0.0],
        trans_lim= 0.0,
        is_train= False,),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter_tracking', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter_tracking', classes=class_names),
    dict(type='ImageNormalize',mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # dict(type='GridMask', use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7, fixed_prob=True),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D_Tracking', classes=class_names),
    dict(type='Collect3D', 
         keys=['img','points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_tracks' , 'gt_futures', 'gt_pasts', 'gt_track_tte'],
         meta_keys=['camera_intrinsics',
                    'camera2ego',
                    'lidar2ego',
                    'lidar2camera',
                    'camera2lidar',
                    'lidar2image',
                    'img_aug_matrix',
                    'lidar_aug_matrix',])
]


modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
gpus=2
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
            dataset_root=data_root,
            ann_file=data_root + tracking_nuscenes_prefix + '_infos_train.pkl',
            pipeline=train_pipeline,
            object_classes=class_names,
            modality=modality,
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
            dataset_root=data_root,
            ann_file=data_root + tracking_nuscenes_prefix + '_infos_val.pkl',
            pipeline=train_pipeline,#test_pipeline,
            object_classes=class_names,
            modality=modality,
            test_mode=False,
            filter_empty_gt=False,
            use_valid_flag=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'))
        )
