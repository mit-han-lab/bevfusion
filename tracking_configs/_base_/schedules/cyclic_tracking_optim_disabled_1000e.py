# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = None
# max_norm=10 is better for SECOND
optimizer_config = None
lr_config = None
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
