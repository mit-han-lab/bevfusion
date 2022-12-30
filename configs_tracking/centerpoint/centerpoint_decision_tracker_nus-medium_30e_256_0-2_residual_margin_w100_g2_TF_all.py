_base_ = [ 
    './centerpoint_decision_tracker_nus-medium_30e_256_0-2_residual_focal_all.py'
]

neptune_tags = ['all dec','nms','focal','TF','w45','FL=5','all classes', '30e', '512 dim']
mm = 0.1
model = dict(
    trk_manager=dict(
            use_det_nms=True,
            is_test_run=False,
            tracked_classes=['car', 'truck', 'bus', 'trailer','motorcycle', 'bicycle','pedestrian'],
            tracker=dict(
                        box_output_mode='track',
                        use_pc_feats=False,
                        use_mp_feats=True,
                        gt_testing=False,
                        cls='All',
                        visualize_cost_mat=False,
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
                        associator=dict(type='TrackingAssociatorMax',use_distance_prior=False,cost_mat_type='margin'),
                        track_supervisor=dict(association=dict(_delete_=True,
                                                                type='MarginAssociationSupervisor',
                                                                balance_supervision=False,
                                                                use_orig_loss=False,
                                                                use_pos=False,
                                                                lambda_pn=1.0,
                                                                lambda_pos=1.0,
                                                                losses={'pn_det_match':dict(type='MarginRankingLoss', margin=mm, reduction='mean'),
                                                                        'pn_det_newborn':dict(type='MarginRankingLoss', margin=mm, reduction='mean'),
                                                                        'pn_track_false_negative':dict(type='MarginRankingLoss', margin=mm, reduction='mean'),
                                                                        'pn_det_false_positive':dict(type='MarginRankingLoss', margin=mm, reduction='mean'),
                                                                        'pn_track_false_positive':dict(type='MarginRankingLoss', margin=mm, reduction='mean'),},
                                                                weights=dict(
                                                                    pn_det_match=10.0,
                                                                    pn_det_false_positive=0.1,
                                                                    pn_track_false_positive=0.1,
                                                                    pn_track_false_negative=10.0,
                                                                    pn_det_newborn=10.0,
                                                                )
                                                            )
                                             ),  
                    ),))


# load_from = 'work_dirs/latest.pth'
# load_from = '/btherien/80_amota_medium_model.pth'
# load_from = "/mnt/bnas/neptune_checkpoints/81.3_amota_focal_nus-medium.pth"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=3,
    train=dict(verbose=False))


