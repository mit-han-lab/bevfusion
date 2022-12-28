#2 encoder layers, 2 deconder layers, 8 heads
_base_ = ['./pnp_net_256_lstm_residual_transformer-0-2-8_decisions_cls_emb.py']


gat_hidden = 128

future_count=6
past_count=4
model=dict(
        net=dict(
            edge_gnn=dict(
                type='GATv2Conv',
                in_channels=gat_hidden,
                out_channels=gat_hidden,
                heads=1,
                concat= False,
                negative_slope= 0.2,
                dropout= 0.0,
                add_self_loops= True,
                edge_dim= None,
                fill_value = 'mean',
                bias= True,
                share_weights= False,
            ),
            decision_edge_linear=dict(type='ModuleDict',
                match=dict(type='Linear', in_features=512, out_features=gat_hidden),
                det_false_positive=dict(type='Linear', in_features=256, out_features=gat_hidden),
                det_newborn=dict(type='Linear', in_features=256, out_features=gat_hidden),
                track_false_positive=dict(type='Linear', in_features=256, out_features=gat_hidden),
                track_false_negative=dict(type='Linear', in_features=256, out_features=gat_hidden),
            ),

            MLPMatch=[dict(type='LinearRes', n_in=gat_hidden, n_out=gat_hidden, norm='GN',ng=32),
                                 dict(type='Linear', in_features=gat_hidden, out_features=1)],

            MLPDetFalsePositive=[dict(type='LinearRes', n_in=gat_hidden, n_out=gat_hidden, norm='GN',ng=64),
                                 dict(type='Linear', in_features=gat_hidden, out_features=1)],
            
            MLPTrackFalsePositive=[dict(type='LinearRes', n_in=gat_hidden, n_out=gat_hidden, norm='GN',ng=64),
                                   dict(type='Linear', in_features=gat_hidden, out_features=1)],
            
            MLPTrackFalseNegative=[dict(type='LinearRes', n_in=gat_hidden, n_out=gat_hidden, norm='GN',ng=64),
                                   dict(type='Linear', in_features=gat_hidden, out_features=1)],

            MLPDetNewborn=[dict(type='LinearRes', n_in=gat_hidden, n_out=gat_hidden, norm='GN',ng=64),
                            dict(type='Linear', in_features=gat_hidden, out_features=1)],

        )
    )