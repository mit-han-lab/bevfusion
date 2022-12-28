#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_256_lstm_residual.py']

future_count=6
past_count=4
model=dict(
        net=dict(
            EncoderNorm=dict(), #dict(type='LayerNorm', normalized_shape=256),
            DecoderNorm=dict(),#type='LayerNorm', normalized_shape=256),
            TransformerEncoderLayer=dict(), #dict(type='TransformerEncoderLayer', d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, activation='relu'),
            TransformerDecoderLayer=dict(),#type='TransformerDecoderLayer', d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, activation='relu'),
            TransformerEncoder=dict(), #dict(type='TransformerEncoder', num_layers=2, norm=None),
            TransformerDecoder=dict(type='LocalFeatureTransformer',d_model=256, nhead=8, layer_names=["self", "cross"] * 4, attention="full",),#type='TransformerDecoder', num_layers=2, norm=None),
            pad=dict(type='Embedding', num_embeddings=1, embedding_dim=256),
        )
    )