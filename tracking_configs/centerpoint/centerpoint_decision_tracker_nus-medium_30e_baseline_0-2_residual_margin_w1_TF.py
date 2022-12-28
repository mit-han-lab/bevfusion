_base_ = [
    '../_base_/tracking_runtime.py',
    '../_base_/datasets/nus-medium-3d-tracking.py',
    '../_base_/trackers/pnp_net_256_lstm_residual_transformer-0-2-8_decisions_bev_mlp.py',
    '../_base_/schedules/cyclic_tracking_30e.py'
]



model = dict(
    net = dict(type='DecisionTracker',
                    merge_forward='interpolate',
                    message_passing_forward='simple',
                    decisions_forward={'det_newborn':'MLP',
                                        'det_false_positive':'MLP',
                                        'track_false_negative':'MLP',
                                        'track_false_positive':'MLP',
                                        'match':'MLP'},),
    trk_manager=dict(
        use_det_nms=True,
            tracker=dict(use_nms=True,
                        tracking_decisions=[],
                        detection_decisions=['det_newborn','det_false_positive'],
                        frameLimit=5,
                        teacher_forcing=True,
                        updater=dict(type='TrackingUpdater',
                                    update_config=dict(
                                        det_newborn=dict(decom=False, output=True, active=True),
                                        det_false_positive=dict(decom=True, output=False, active=False),
                                        track_false_negative=dict(decom=False, output=True, active=True),
                                        track_false_positive=dict(decom=True, output=False, active=False),
                                        track_unmatched=dict(decom=False, output=False, active=True),
                                        match=dict(decom=False, output=True, active=True),
                                    )),
                        associator=dict(type='TrackingAssociator',use_distance_prior=False,cost_mat_type='softmax'),
                        track_supervisor=dict(association=dict(type='MarginAssociationSupervisor',
                                                                balance_supervision=False,
                                                                use_orig_loss=False,
                                                                use_pos=True,
                                                                compute_summary=True),
                                                                losses={
                                                                    'det_match-det_newborn':dict(type='MarginRankingLoss', margin=0.2, reduction='mean'),
                                                                    'det_match-det_false_positive':dict(type='MarginRankingLoss', margin=0.4, reduction='mean'),
                                                                    'det_newborn-det_false_positive':dict(type='MarginRankingLoss', margin=0.2, reduction='mean'),
                                                                    'pn_det_match':dict(type='MarginRankingLoss', margin=0.6, reduction='mean'),
                                                                    'pn_det_newborn':dict(type='MarginRankingLoss', margin=0.4, reduction='mean'),
                                                                    'pn_det_false_positive':dict(type='MarginRankingLoss', margin=0.2, reduction='mean'),
                                                                    },
                        )
                    ),)

)               

neptune_tags = ['baseline','nms','focal','w=25','TF']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

custom_hooks = [
    dict(type='CustomEval', priority='NORMAL', interval=10, eval_at_zero=False, eval_start_epoch=0),
    dict(type='ShuffleDatasetHook', priority='NORMAL'),
    dict(type='SaveModelToNeptuneHook', priority=40),
    dict(type='SetEpochInfoHook', priority=1),
]

