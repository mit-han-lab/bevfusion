_base_ = [ 
    './centerpoint_decision_tracker_nus-medium_30e_0-2_residual_focal.py'
]

mm = 0.2 #margin multiplier


model = dict(
    trk_manager=dict(
            is_test_run=False,
            tracker=dict(visualize_cost_mat=False,
                        use_nms=True,
                        frameLimit=5,
                        teacher_forcing=True,
                        updater=dict(type='TrackingUpdater',
                                    update_config=dict(
                                        det_newborn=dict(decom=False, output=True, active=True),
                                        det_false_positive=dict(decom=False, output=False, active=True),
                                        track_false_negative=dict(decom=False, output=True, active=True),
                                        track_false_positive=dict(decom=True, output=False, active=False),
                                        track_unmatched=dict(decom=True, output=False, active=False),
                                        match=dict(decom=False, output=True, active=True),
                                    )),
                        associator=dict(type='TrackingAssociatorMax',use_distance_prior=False,cost_mat_type='margin'),
                        track_supervisor=dict(association=dict(_delete_=True,
                                                                type='MarginAssociationSupervisor',
                                                                balance_supervision=False,
                                                                use_orig_loss=False,
                                                                use_pos=True,
                                                                lambda_pn=0.01,
                                                                lambda_pos=0.01,
                                                                losses={'pn_det_match':dict(type='MarginRankingLoss', margin=3*mm, reduction='sum'),
                                                                        'pn_det_newborn':dict(type='MarginRankingLoss', margin=2*mm, reduction='sum'),
                                                                        'pn_track_false_negative':dict(type='MarginRankingLoss', margin=2*mm, reduction='sum'),
                                                                        'pn_det_false_positive':dict(type='MarginRankingLoss', margin=1*mm, reduction='sum'),
                                                                        'pn_track_false_positive':dict(type='MarginRankingLoss', margin=1*mm, reduction='sum'),

                                                                        'det_match-det_newborn':dict(type='MarginRankingLoss', margin=1*mm, reduction='sum'),
                                                                        'det_match-det_false_positive':dict(type='MarginRankingLoss', margin=2*mm, reduction='sum'),
                                                                        'det_newborn-det_false_positive':dict(type='MarginRankingLoss', margin=1*mm, reduction='sum'),

                                                                        'det_match-track_false_negative':dict(type='MarginRankingLoss', margin=1*mm, reduction='sum'),
                                                                        'det_match-track_false_positive':dict(type='MarginRankingLoss', margin=2*mm, reduction='sum'),
                                                                        'track_false_negative-track_false_positive':dict(type='MarginRankingLoss', margin=1*mm, reduction='sum'),},
                                                                compute_summary=True,
                                                                weights={
                                                                    'pn_det_match':2.0,
                                                                    'pn_det_false_positive':1.0,
                                                                    'pn_track_false_positive':1.0,
                                                                    'pn_track_false_negative':10.0,
                                                                    'pn_det_newborn':10.0,}
                                                                ),)
                        )
                    ),)





data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(verbose=False))
