_base_ = [
    '../_base_/tracking_runtime.py',
    '../_base_/datasets/nus-medium-3d-tracking-bevfusion.py',
    '../_base_/trackers/pnp_net_256_lstm_residual_transformer-0-2-8_decisions_cls_emb.py',
    '../_base_/schedules/cyclic_tracking_30e_accum2.py'
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
                        tracking_decisions=['track_false_negative','track_false_positive'],
                        detection_decisions=['det_newborn','det_false_positive'],
                        frameLimit=5,
                        teacher_forcing=True,
                        propagation_method='velocity',
                        updater=dict(type='TrackingUpdater',
                                    update_config=dict(
                                        det_newborn=dict(decom=False, output=True, active=True),
                                        det_false_positive=dict(decom=False, output=False, active=True),
                                        track_false_negative=dict(decom=False, output=True, active=True),
                                        track_false_positive=dict(decom=True, output=False, active=False),
                                        track_unmatched=dict(decom=True, output=False, active=False),
                                        match=dict(decom=False, output=True, active=True),
                                    )),
                        associator=dict(type='TrackingAssociatorMax',use_distance_prior=False,cost_mat_type='softmax'),
                        track_supervisor=dict(association=dict(type='FocalLossAssociationSupervisor',
                                                                focal_loss=dict(type='FocalLoss',reduction='mean',gamma=2.0),
                                                                l1_loss=dict(type='SmoothL1Loss',reduction='sum'),
                                                                compute_summary=True,
                                                                weights=dict(
                                                                    match=25.0,
                                                                    det_false_positive=1.0,
                                                                    track_false_positive=1.0,
                                                                    track_false_negative=25.0,
                                                                    det_newborn=25.0,
                                                                )),)
                        )
                    ),)

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=4,
# )

custom_hooks = [
    dict(type='CustomEval', priority='NORMAL', interval=31, eval_at_zero=False, eval_start_epoch=0),
    dict(type='ShuffleDatasetHook', priority='NORMAL'),
    dict(type='SaveModelToNeptuneHook', priority=40),
    dict(type='SetEpochInfoHookTracking', priority=1),
]

# custom_hooks += [ dict(type='DebugPrintingHook', priority=1+x*10)for x in range(10)]


# custom_hooks.0.eval_at_zero=True
