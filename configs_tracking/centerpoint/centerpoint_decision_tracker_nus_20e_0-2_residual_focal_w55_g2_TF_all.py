_base_ = [ 
    './centerpoint_decision_tracker_nus_20e_0-2_residual_focal_all.py'
]

neptune_tags = ['all dec','nms','focal','TF','w45','all', 'FL=5']

model = dict(
    trk_manager=dict(
            use_det_nms=True,
            is_test_run=False,
            tracked_classes=['car', 'truck', 'bus', 'trailer','motorcycle', 'bicycle','pedestrian'],
            tracker=dict(
                        cls='All',
                        visualize_cost_mat=False,
                        use_mp_feats=True,
                        gt_testing=False,
                        use_nms=True,
                        frameLimit=5,
                        teacher_forcing=True,
                        propagation_method='future',
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
                                                                    match=55.0,
                                                                    det_false_positive=1.0,
                                                                    track_false_positive=1.0,
                                                                    track_false_negative=55.0,
                                                                    det_newborn=55.0,
                                                                )),)
                        )
                    ),)


# load_from = 'work_dirs/latest.pth'
# load_from = '/btherien/80_amota_medium_model.pth'
# load_from = "/mnt/bnas/neptune_checkpoints/81.3_amota_focal_nus-medium.pth"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(verbose=False))




