#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_256_lstm_residual_transformer-0-2-8_decisions.py']

model=dict(
        net=dict(
            class_embeddings = dict(type='Embedding', num_embeddings=10, embedding_dim=256),
            synthetic_query = dict(type='Linear', in_features=256, out_features=128),
        )
    )