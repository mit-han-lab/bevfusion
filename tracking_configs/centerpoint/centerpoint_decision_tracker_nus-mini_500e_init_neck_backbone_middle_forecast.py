_base_ = [
    './centerpoint_decision_tracker_nus-mini_500e_0-2_residual_focal.py'
]

model = dict(use_neck=dict(use=True,
                           init='new',
                           pts_neck=dict(
                                        type='SECONDFPN',
                                        in_channels=[128, 256],
                                        out_channels=[256, 256],
                                        upsample_strides=[1, 2],
                                        norm_cfg=dict(type='GN', num_groups=8),
                                        upsample_cfg=dict(type='deconv', bias=False),
                                        use_conv_for_no_stride=True),),
            use_backbone=dict(use=True,
                            init='new',
                            pts_backbone=dict(
                                            type='SECOND_GN',
                                            in_channels=256,
                                            out_channels=[128, 256],
                                            layer_nums=[5, 5],
                                            layer_strides=[1, 2],
                                            norm_cfg=[dict(type='GN', num_groups=8),dict(type='GN', num_groups=16)],
                                            conv_cfg=dict(type='Conv2d', bias=False)),),

            use_middle_encoder=dict(use=True,
                                    init='new',
                                    pts_middle_encoder=dict(
                                        type='SparseEncoderGN',
                                        in_channels=5,
                                        sparse_shape=[41, 1024, 1024],
                                        output_channels=128,
                                        norm_cfg=dict(type='GN', num_groups=8),
                                        order=('conv', 'norm', 'act'),
                                        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                                                    128)),
                                        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                                        block_type='basicblock'),

                                    bev_supervisor=dict(use_metric=False,))
)


