# _base_ = ['mmpose::_base_/default_runtime.py']
_base_ = ['../../../configs/_base_/default_runtime.py']
import os
import sys
# sys.path.append('/workspace/mmpose3d/projects/rtmpose3d')
custom_imports = dict(imports=['rtmpose3d'], allow_failed_imports=False)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizerPlus', vis_backends=vis_backends, name='visualizer')

# runtime
max_epochs = 270
stage2_num_epochs = 10
base_lr = 5e-4
num_keypoints = 133

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=2024)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=4096)

# codec settings
codec = dict(
    type='SimCC3DLabel',
    input_size=(288, 384, 288),
    sigma=(6., 6.93, 6.),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    root_index=(11, 12))

backbone_path = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'  # noqa

# model settings
model = dict(
    type='TopdownPoseEstimator3D',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=backbone_path)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    head=dict(
        type='RTMW3DHead',
        in_channels=1024,
        out_channels=133,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=[
            dict(
                type='KLDiscretLossWithWeight',
                use_target_weight=True,
                beta=10.,
                label_softmax=True),
            dict(
                type='BoneLoss',
                joint_parents=[
                    0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30,
                    31, 32, 33, 34, 35, 36, 37, 38, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                    3, 50, 50, 51, 52, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3,
                    3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 7, 91, 92, 93, 94, 91, 96, 97, 98, 91, 100,
                    101, 102, 91, 104, 105, 106, 91, 108, 109, 110, 8, 112,
                    113, 114, 113, 112, 117, 118, 117, 112, 121, 122, 123, 112,
                    125, 126, 127, 112, 129, 130, 131
                ],
                use_target_weight=True,
                loss_weight=2.0)
        ],
        decoder=codec),
    # test_cfg=dict(flip_test=False, mode='2d')
    test_cfg=dict(flip_test=False))

# base dataset settings
data_mode = 'topdown'
dataset_type = 'H36MWholeBodyDataset'
backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# mapping

data_mode = 'topdown'
data_root = 'data/'



map_uepose_to_coco = [
    [0 ,0 ],
    [1 ,1 ],
    [2 ,2 ],
    [3 ,3 ],
    [4 ,4 ],
    [5 ,5 ],
    [6 ,6 ],
    [8 ,8 ],
    [9 ,9 ],
    [10,10],
    [11,11],
    [12,12],
    [13,13],
    [14,14],
    [15,15],
    [16,16],
]


uepose_dataset = dict(
        type='UnrealPose3dDataset',
        data_root='/workspace/MobileHumanPose3D/dataset/uecoco_3d',
        ann_file='annotations/test.json',
        data_mode='topdown',
        causal=True,
        seq_len=1,
        data_prefix=dict(img='test/'),
        subset_frac=0.1,
        pipeline=[
            dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=map_uepose_to_coco,
        ),
    ],
    metainfo= dict(from_file='/workspace/mmpose3d/configs/_base_/datasets/uepose.py'))




# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        datasets=[uepose_dataset],
        pipeline=train_pipeline,
        metainfo=dict(from_file='/workspace/mmpose3d/configs/_base_/datasets/h3wb.py'),
        test_mode=False)
    )
# hooks
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1))




val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
     dataset=dict(
        type='CombinedDataset',
        datasets=[uepose_dataset],
        pipeline=val_pipeline,
        metainfo=dict(from_file='/workspace/mmpose3d/configs/_base_/datasets/h3wb.py'),
        test_mode=False))

test_dataloader = val_dataloader
# evaluators
val_evaluator = [
    dict(type='SimpleMPJPE', mode='mpjpe'),
    dict(type='SimpleMPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator

