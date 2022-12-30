"""File make to change the config of nms for tracking"""

_base_ = ['./centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus.py']

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
            final_kernel=3)),
    test_cfg=dict(pts=dict(nms_type='circle',))) #min_radius=[5, 12, 10, 1, 0.85, 0.175])))


#load_from='/btherien/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth'