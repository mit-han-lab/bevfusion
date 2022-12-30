_base_ = [
    "../_base_/datasets/nus-reidentificaiton.py",
    "../_base_/reidentifiers/reid_base.py",
    "../_base_/schedules/cyclic_100e_lr1e-5.py",
    "../_base_/reidentification_runtime.py",
]



neptune_tags = ['100e']