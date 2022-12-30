_base_ = ['./pnp_net_512_lstm_bev-interpolate_transformer-3-3-8.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPFalsePositive=[dict(type='Linear', in_features=512, out_features=256),
                                dict(type='GroupNorm', num_groups=16, num_channels=256),
                                dict(type='ReLU'),
                                dict(type='Linear', in_features=256, out_features=128),
                                dict(type='GroupNorm', num_groups=8, num_channels=128),
                                dict(type='ReLU'),
                                dict(type='Linear', in_features=128, out_features=1)],
        )
    )