_base_ = [ 
    './centerpoint_decision_tracker_nus-medium_30e_256_0-2_residual_focal_all_bevfusion.py'
]

neptune_tags = ['GT Testing', 'confidence_threshold=0.004']#['all dec','nms','focal','TF','w45','FL=5','all classes', '30e','conf filter 0.000001']
mm = 0.1
model = dict(
    confidence_threshold=0.004, #0.002 (previous best) (0.004 --> 0.923 amota)
    trk_manager=dict(
            use_det_nms=True,
            is_test_run=True,
            tracked_classes=['car', 'truck', 'bus', 'trailer','motorcycle', 'bicycle','pedestrian'],
            tracker=dict(
                        box_output_mode='track',
                        use_pc_feats=False,
                        use_mp_feats=True,
                        gt_testing=True,
                        cls='All',
                        visualize_cost_mat=False,
                        use_nms=True,
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
                                                                loss_weights=dict(
                                                                    track=1.0,
                                                                    det=1.0,
                                                                    consistency=1.0,
                                                                    assemble=1.0,
                                                                ),
                                                                weights=dict(
                                                                    match=45.0,
                                                                    det_false_positive=1.0,
                                                                    track_false_positive=1.0,
                                                                    track_false_negative=45.0,
                                                                    det_newborn=45.0,
                                                                )),
                                                            
                                             ),  
                    ),))


load_from = 'runs/run/latest.pth'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=3,
    train=dict(verbose=False))


