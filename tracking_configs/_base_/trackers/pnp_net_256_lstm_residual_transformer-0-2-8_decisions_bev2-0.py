#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_256_lstm_residual_transformer-0-2-8.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            MLPDetFalsePositive=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=16),
                                 dict(type='Linear', in_features=256, out_features=1)],
            
            MLPTrackFalsePositive=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=16),
                                   dict(type='Linear', in_features=256, out_features=1)],
            
            MLPTrackFalseNegative=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=16),
                                   dict(type='Linear', in_features=256, out_features=1)],

            MLPDetNewborn=[dict(type='LinearRes', n_in=256, n_out=256, norm='GN',ng=16),
                            dict(type='Linear', in_features=256, out_features=1)],

            decisions=dict(embedding=dict(type='Embedding', num_embeddings=1, embedding_dim=256),
                           det_false_positive=dict(),
                           track_false_positive=dict(),
                           track_false_negative=dict(),
                           det_newborn=dict(),),
            false_negative_emb=dict(type='Embedding', num_embeddings=1, embedding_dim=256),



            bev_forecast_proj=[dict(type='Linear', in_features=128, out_features=256),
                               dict(type='GroupNorm', num_groups=16, num_channels=256),
                               dict(type='ReLU')],


            bev_confidence_proj=[dict(type='LinearRes', n_in=128, n_out=128, norm='GN',ng=8),
                                 dict(type='Linear', in_features=128, out_features=1),],

            bev_pos_enc=dict(type='Embedding', num_embeddings=6, embedding_dim=512),
            bev_EncoderNorm=dict(type='LayerNorm', normalized_shape=512),
            bev_TransformerEncoderLayer=dict(type='TransformerEncoderLayer', d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu'),
            bev_TransformerEncoder=dict(type='TransformerEncoder', num_layers=1, norm=None),
        )
    )