num_classes  = 10
num_points = 2048
hidden_size = 256
ng = 32


model = dict(
    type='ReIDNet',
    hidden_size=hidden_size,
    backbone=dict(type='Pointnet_Backbone',input_channels=0,use_xyz=True,),
    cls_head=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=ng),
              dict(type='Linear', in_features=512, out_features=num_classes)],

    match_head=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=ng),
                dict(type='Linear', in_features=512, out_features=1)],
    shape_head=[
        dict(type='Conv1d', in_channels=512, out_channels=1024, kernel_size=12),
        dict(type='BatchNorm1d', num_features=1024),
        dict(type='ReLU'),
        dict(type='Conv1d', in_channels=1024, out_channels=2048, kernel_size=10),
        dict(type='BatchNorm1d', num_features=2048),
        dict(type='ReLU'),
        dict(type='Conv1d', in_channels=2048, out_channels=2048, kernel_size=10),
    ],
    cross_stage1=dict(type='corss_attention',d_model=32,nhead=2,attention='linear'),
    local_stage1=dict(type='local_self_attention',d_model=32,nhead=2,attention='linear',knum=48),
    cross_stage2=dict(type='corss_attention',d_model=32,nhead=2,attention='linear'),
    local_stage2=dict(type='local_self_attention',d_model=32,nhead=2,attention='linear',knum=48),
)


                 