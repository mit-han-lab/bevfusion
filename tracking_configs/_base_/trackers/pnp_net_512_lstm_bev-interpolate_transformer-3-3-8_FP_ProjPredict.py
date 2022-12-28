_base_ = ['./pnp_net_512_lstm_bev-interpolate_transformer-3-3-8_FP.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPProjPredict=[dict(type='Linear', in_features=512, out_features=512),
                       dict(type='GroupNorm', num_groups=32, num_channels=512),
                       dict(type='ReLU')],
        )
    )