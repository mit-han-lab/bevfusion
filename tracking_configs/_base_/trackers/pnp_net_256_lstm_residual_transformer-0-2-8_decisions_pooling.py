#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_256_lstm_residual_transformer-0-2-8.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPDetFalsePositive=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=32),
                                 dict(type='Linear', in_features=512, out_features=1)],
            
            MLPTrackFalsePositive=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=32),
                                   dict(type='Linear', in_features=512, out_features=1)],
            
            MLPTrackFalseNegative=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=32),
                                   dict(type='Linear', in_features=512, out_features=1)],

            MLPDetNewborn=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=32),
                            dict(type='Linear', in_features=512, out_features=1)],

            decision_pooling=dict(det_newborn=dict(type='GatedPooling', node_dim=512, pooling_dim=512),
                                  det_false_positive=dict(type='GatedPooling', node_dim=512, pooling_dim=512),
                                  track_false_positive=dict(type='GatedPooling', node_dim=512, pooling_dim=512),
                                  track_false_negative=dict(type='GatedPooling', node_dim=512, pooling_dim=512),
                                ),
            self_loop_pooling=dict(det_newborn=dict(type='Linear', in_features=256, out_features=512),
                                  det_false_positive=dict(type='Linear', in_features=256, out_features=512),
                                  track_false_positive=dict(type='Linear', in_features=256, out_features=512),
                                  track_false_negative=dict(type='Linear', in_features=256, out_features=512),
                                ),

            false_negative_emb=dict(type='Embedding', num_embeddings=1, embedding_dim=256),
        )
    )