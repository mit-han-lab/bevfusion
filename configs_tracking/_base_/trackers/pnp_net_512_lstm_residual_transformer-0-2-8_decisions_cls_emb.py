#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_512_lstm_residual.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            EncoderNorm=dict(), #dict(type='LayerNorm', normalized_shape=256),
            DecoderNorm=dict(type='LayerNorm', normalized_shape=512),
            TransformerEncoderLayer=dict(), #dict(type='TransformerEncoderLayer', d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, activation='relu'),
            TransformerDecoderLayer=dict(type='TransformerDecoderLayer', d_model=512, nhead=16, dim_feedforward=1024, dropout=0.1, activation='relu'),
            TransformerEncoder=dict(), #dict(type='TransformerEncoder', num_layers=2, norm=None),
            TransformerDecoder=dict(type='TransformerDecoder', num_layers=2, norm=None),
            pad=dict(type='Embedding', num_embeddings=1, embedding_dim=512),


            MLPDetFalsePositive=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=64),
                                 dict(type='Linear', in_features=512, out_features=1)],
            
            MLPTrackFalsePositive=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=64),
                                   dict(type='Linear', in_features=512, out_features=1)],
            
            MLPTrackFalseNegative=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=64),
                                   dict(type='Linear', in_features=512, out_features=1)],

            MLPDetNewborn=[dict(type='LinearRes', n_in=512, n_out=512, norm='GN',ng=64),
                            dict(type='Linear', in_features=512, out_features=1)],

            false_negative_emb=dict(type='Embedding', num_embeddings=1, embedding_dim=512),
            class_embeddings = dict(type='Embedding', num_embeddings=10, embedding_dim=512),
        )
    )