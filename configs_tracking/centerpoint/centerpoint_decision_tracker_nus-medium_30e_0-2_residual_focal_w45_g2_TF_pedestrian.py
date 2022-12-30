_base_ = [ 
    './centerpoint_decision_tracker_nus-medium_30e_0-2_residual_focal.py'
]

model = dict(
    trk_manager=dict(
            is_test_run=False,
            tracked_classes=['pedestrian'],
            tracker=dict(visualize_cost_mat=False,
                        use_mp_feats=False,
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
                        associator=dict(type='TrackingAssociatorMax',use_distance_prior=False,cost_mat_type='softmax'),
                        track_supervisor=dict(association=dict(type='FocalLossAssociationSupervisor',
                                                                focal_loss=dict(type='FocalLoss',reduction='mean',gamma=2.0),
                                                                l1_loss=dict(type='SmoothL1Loss',reduction='sum'),
                                                                compute_summary=True,
                                                                weights=dict(
                                                                    match=45.0,
                                                                    det_false_positive=1.0,
                                                                    track_false_positive=1.0,
                                                                    track_false_negative=45.0,
                                                                    det_newborn=45.0,
                                                                )),)
                        )
                    ),)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=3)






