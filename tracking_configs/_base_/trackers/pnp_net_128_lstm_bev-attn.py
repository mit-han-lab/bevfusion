_base_ = ['./pnp_net_128_lstm.py']


model=dict(
        net=dict(
            bev_attn=dict(type='MultiheadAttention', embed_dim=64, num_heads=2),

            MLP_encode_BEV2=[dict(type='Linear', in_features=64, out_features=32), #key 
                             dict(type='ReLU')],
        )
    )