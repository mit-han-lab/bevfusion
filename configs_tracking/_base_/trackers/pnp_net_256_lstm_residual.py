_base_ = ['./pnp_net.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            lstm=dict(type='LSTM',input_size=256, hidden_size=256, num_layers=1),

            MLP_encode_BEV1=[dict(type='Linear', in_features=512, out_features=128),
                             dict(type='GroupNorm', num_groups=8, num_channels=128),
                             dict(type='ReLU'),],

            MLP_encode_motion=[dict(type='Linear', in_features=16, out_features=128),
                               dict(type='GroupNorm', num_groups=8, num_channels=128),
                               dict(type='ReLU')], 

            MLPMerge=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=32),],

            MLPMatch=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=64),
                      dict(type='Linear', in_features=512, out_features=1)],

            MLPRefine=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=1),
                        dict(type='Linear', in_features=256, out_features=1+(1+past_count)*2)],

            MLPPredict=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=1),
                        dict(type='Linear', in_features=256, out_features=future_count*2)],

            h0=dict(type='Embedding', num_embeddings=1, embedding_dim=256),
            c0=dict(type='Embedding', num_embeddings=1, embedding_dim=256),
        )
    )




