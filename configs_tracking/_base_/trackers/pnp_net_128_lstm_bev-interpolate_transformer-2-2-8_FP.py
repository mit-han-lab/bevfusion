_base_ = ['./pnp_net_128_lstm_bev-interpolate_transformer.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPFalsePositive=[dict(type='Linear', in_features=128, out_features=128),
                      dict(type='GroupNorm', num_groups=8, num_channels=128),
                      dict(type='ReLU'),
                      dict(type='Linear', in_features=128, out_features=64),
                      dict(type='GroupNorm', num_groups=4, num_channels=64),
                      dict(type='ReLU'),
                      dict(type='Linear', in_features=64, out_features=1)],
        )
    )