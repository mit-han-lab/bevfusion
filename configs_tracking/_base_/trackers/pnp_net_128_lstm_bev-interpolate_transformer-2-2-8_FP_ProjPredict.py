_base_ = ['./pnp_net_128_lstm_bev-interpolate_transformer_FP.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPProjPredict=[dict(type='Linear', in_features=512, out_features=128),
                       dict(type='GroupNorm', num_groups=8, num_channels=128),
                       dict(type='ReLU')],
        )
    )