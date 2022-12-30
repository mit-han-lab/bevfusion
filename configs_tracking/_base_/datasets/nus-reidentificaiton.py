train_pipeline = [
]

tracking_classes = {'vehicle.bicycle':'bicycle',
                    'vehicle.bus.bendy':'bus',
                    'vehicle.bus.rigid':'bus',
                    'vehicle.car':'car',
                    'vehicle.motorcycle':'motorcycle',
                    'human.pedestrian.adult':'pedestrian',
                    'human.pedestrian.child':'pedestrian',
                    'human.pedestrian.construction_worker':'pedestrian',
                    'human.pedestrian.police_officer':'pedestrian',
                    'vehicle.trailer':'trailer',
                    'vehicle.truck':'truck'}

cls_to_idx = {
    'none_key':-1,
    'car':0, 'truck':1, 
    'construction_vehicle':2, 'bus':3, 
    'trailer':4, 'barrier':5, 
    'motorcycle':6, 'bicycle':7, 
    'pedestrian':8, 'traffic_cone':9
}

version = 'medium'
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(type='ReIDDataset',
               train=True,
               metadata_path_complete='data/lstk/complete/nuscenes/metadata.pkl', 
               metadata_path_sparse='data/lstk/sparse-{}/metadata/metadata.pkl'.format(version),
               data_root_complete='data/lstk/complete/nuscenes',
               data_root_sparse='data/lstk/sparse-{}'.format(version),
               load_scene=True,
               load_objects=True,
               load_feats=['xyz'],
               load_dims=[3],
               to_ego_frame=False,
               min_points=2,
               cls_to_idx=cls_to_idx,
               tracking_classes=tracking_classes,
               return_mode='dict'),
    val=dict(type='ReIDDataset',
               train=True,
               metadata_path_complete='data/lstk/complete/nuscenes/metadata.pkl', 
               metadata_path_sparse='data/lstk/sparse-{}/metadata/metadata.pkl'.format(version),
               data_root_complete='data/lstk/nuscenes/complete',
               data_root_sparse='data/lstk/sparse-{}'.format(version),
               load_scene=True,
               load_objects=True,
               load_feats=['xyz'],
               load_dims=[3],
               to_ego_frame=False,
               min_points=2,
               cls_to_idx=cls_to_idx,
               tracking_classes=tracking_classes,
               return_mode='dict'),
)

