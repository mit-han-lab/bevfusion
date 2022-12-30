_base_ = ['./pnp_net_128_lstm.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            EncoderNorm=dict(type='LayerNorm', normalized_shape=128),
            DecoderNorm=dict(type='LayerNorm', normalized_shape=128),
            TransformerEncoderLayer=dict(type='TransformerEncoderLayer', d_model=128, nhead=8, dim_feedforward=256, dropout=0.1, activation='relu'),
            TransformerDecoderLayer=dict(type='TransformerDecoderLayer', d_model=128, nhead=8, dim_feedforward=256, dropout=0.1, activation='relu'),
            TransformerEncoder=dict(type='TransformerEncoder', num_layers=2, norm=None),
            TransformerDecoder=dict(type='TransformerDecoder', num_layers=2, norm=None),
            pad=dict(type='Embedding', num_embeddings=1, embedding_dim=128),
        )
    )