#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_128_lstm_residual_transformer-0-2-8.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPDetFalsePositive=[dict(type='LinearRes', n_in=128, n_out=128, norm='GN',ng=8),
                                 dict(type='Linear', in_features=128, out_features=1)],
            
            MLPTrackFalsePositive=[dict(type='LinearRes', n_in=128, n_out=128, norm='GN',ng=8),
                                   dict(type='Linear', in_features=128, out_features=1)],
            
            MLPTrackFalseNegative=[dict(type='LinearRes', n_in=128, n_out=128, norm='GN',ng=8),
                                   dict(type='Linear', in_features=128, out_features=1)],

            MLPDetNewborn=[dict(type='LinearRes', n_in=128, n_out=128, norm='GN',ng=8),
                            dict(type='Linear', in_features=128, out_features=1)],

            decisions=dict(embedding=dict(type='Embedding', num_embeddings=1, embedding_dim=128),
                           det_false_positive=dict(),
                           track_false_positive=dict(),
                           track_false_negative=dict(),
                           det_newborn=dict(),),
            false_negative_emb=dict(type='Embedding', num_embeddings=1, embedding_dim=128),
        )
    )