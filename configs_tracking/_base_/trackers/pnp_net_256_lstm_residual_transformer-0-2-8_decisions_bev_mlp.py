#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_256_lstm_residual_transformer-0-2-8.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPDetFalsePositive=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=32),
                                 dict(type='Linear', in_features=256, out_features=1)],
            
            MLPTrackFalsePositive=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=32),
                                   dict(type='Linear', in_features=256, out_features=1)],
            
            MLPTrackFalseNegative=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=32),
                                   dict(type='Linear', in_features=256, out_features=1)],

            MLPDetNewborn=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=32),
                            dict(type='Linear', in_features=256, out_features=1)],

            decisions=dict(embedding=dict(type='Embedding', num_embeddings=1, embedding_dim=256),
                           det_false_positive=dict(),
                           track_false_positive=dict(),
                           track_false_negative=dict(),
                           det_newborn=dict(),),
            false_negative_emb=dict(type='Embedding', num_embeddings=1, embedding_dim=256),

            MLP_encode_BEV1=[dict(type='LinearRes', n_in=2560, n_out=512, norm='GN',ng=32),
                             dict(type='Linear', in_features=512, out_features=256),
                             dict(type='GroupNorm', num_groups=16, num_channels=256),
                             dict(type='ReLU'),
                             dict(type='Linear', in_features=256, out_features=128),
                             dict(type='GroupNorm', num_groups=8, num_channels=128),
                             dict(type='ReLU'),
                             dict(type='Linear', in_features=128, out_features=128),
                             dict(type='GroupNorm', num_groups=8, num_channels=128),
                             dict(type='ReLU'),],
        )
    )