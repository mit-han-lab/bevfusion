_base_ = ['./centerpoint_0075voxel_second_secfpn_gn_4x8_cyclic_20e_nus.py']

model = dict(
    pts_bbox_head=dict(
        separate_head=dict(
            type='DCNSeparateHead',
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4),
            init_bias=-2.19,
            final_kernel=3,
            norm_cfg=dict(type='GN', num_groups=4),
            )
            ),
    test_cfg=dict(pts=dict(nms_type='circle')))

data = dict(
    samples_per_gpu=6)
#load_from='/btherien/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth'