_base_ = ['./default_runtime.py']


workflow = [('train', 1)]#,('val',1)]
work_dir='work_dirs'

checkpoint_config = dict(interval=1,max_keep_ckpts=1,save_last=True,save_optimizer=True)
evaluation = dict(interval=100000, pipeline=[])
find_unused_parameters=True

cudnn_benchmark=False
dataloader_kwargs = dict(shuffle=False, prefetch_factor=10)
train_tracker = False
seed=42
deterministic=False

log_config = dict(interval=16,
                hooks=[
                    dict(type='TextLoggerHook',reset_flag=True),
                    dict(type="NeptuneLoggerHook",
                        init_kwargs={    
                            'project':"bentherien/Tracking",
                            'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNWQ4NTRlNi00ZGJlLTRhZDctYmRlOC0zZWM4NmE3YWE0MWIifQ==",
                            'name': "",
                            'source_files': [],
                            'tags': [] 
                            },
                        interval=16,
                        ignore_last=True,
                        reset_flag=True,
                        with_step=True,
                        by_epoch=True)
            ])
