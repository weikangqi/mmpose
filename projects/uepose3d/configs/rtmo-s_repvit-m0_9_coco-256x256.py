_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(imports=['uepose'], allow_failed_imports=False)
# runtime
train_cfg = dict(max_epochs=1000, val_interval=10, dynamic_intervals=[(580, 1)])

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=3))

optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({'neck.encoder': dict(lr_mult=0.05)})),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=5,
        T_max=280,
        end=280,
        by_epoch=True,
        convert_to_iter_based=True),
    # this scheduler is used to increase the lr from 2e-4 to 5e-4
    dict(type='ConstantLR', by_epoch=True, factor=2.5, begin=280, end=281),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=281,
        T_max=300,
        end=580,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=580, end=600),
]

# data
input_size = (256, 256)
metafile = '/workspace/mmpose3d/configs/_base_/datasets/uepose.py'
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=input_size,
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=input_size,
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=input_size,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=input_size,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

data_mode = 'bottomup'
data_root = 'data/'

# small_prefix='small_'
small_prefix=''

# train datasets
coco_train_dataset = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file=f'coco/annotations/{small_prefix}person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    # pipeline=train_pipeline_stage1,
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=21,
            mapping=[(i, i) for i in range(17)])
    ],
)

coco_val_dataset = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file=f'coco/annotations/{small_prefix}person_keypoints_val2017.json',
    data_prefix=dict(img='coco/val2017/'),
    pipeline=[
        dict(
        type='KeypointConverter',
        num_keypoints=21,
        mapping=[(i, i) for i in range(17)])
    ],
)


mpii_uepose = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (6, 20),
    (9, 19),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
    
]

mpii_dataset = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='mpii/images/'),
    pipeline=[
        dict(type='KeypointConverter', num_keypoints=21, mapping=mpii_uepose)
    ],
)


# ochuman_dataset = dict(
#     type='CocoDataset',
#     data_root=data_root,
#     data_mode=data_mode,
#     ann_file='ochuman/annotations/mpii_train.json',
#     data_prefix=dict(img='ochuman/images/'),
#     pipeline=[
#         dict(type='KeypointConverter', num_keypoints=21, mapping=mpii_uepose)
#     ],
# )




uepose_val_dataset = dict(
    type='UnrealPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='uecoco/annotations/test.json',
    data_prefix=dict(img='uecoco/test/'),
    pipeline = []
)



train_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file=metafile),
    datasets=[
        coco_train_dataset,
        mpii_dataset,
    ],
    sample_ratio_factor=[1,1],
    test_mode=False,
    pipeline=train_pipeline_stage1)



train_dataloader = dict(
    batch_size=24,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset = dict(
        type='CombinedDataset',
        metainfo=dict(from_file=metafile),
        datasets=[
            coco_val_dataset,
        ],
        sample_ratio_factor=[1],
        test_mode=True,
        pipeline=val_pipeline)
    )
   
test_dataloader = val_dataloader

# evaluators

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file='/workspace/MobileHumanPose3D/dataset/uecoco/annotations/test.json',
#     score_mode='bbox',
#     nms_mode='none'
# )

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'/workspace/mmpose3d/data/coco/annotations/{small_prefix}person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
    gt_converter= dict(
        type='KeypointConverter',
        num_keypoints=21,
        mapping=[(i, i) for i in range(17)])
)


test_evaluator = val_evaluator

# hooks
custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=20,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(
        type='RTMOModeSwitchHook',
        epoch_attributes={
            280: {
                'proxy_target_cc': True,
                'overlaps_power': 1.0,
                'loss_cls.loss_weight': 2.0,
                'loss_mle.loss_weight': 5.0,
                'loss_oks.loss_weight': 10.0
            },
        },
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

# model
widen_factor = 0.5
deepen_factor = 0.33

model = dict(
    type='BottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1),
        ]),
    backbone=dict(
        type='RepViT',
          cfgs = [
           [3,   2,  48, 1, 0, 1],
            [3,   2,  48, 0, 0, 1],
            [3,   2,  48, 0, 0, 1],
            [3,   2,  96, 0, 0, 2],
            [3,   2,  96, 1, 0, 1],
            [3,   2,  96, 0, 0, 1],
            [3,   2,  96, 0, 0, 1],
            [3,   2,  192, 0, 1, 2],
            [3,   2,  192, 1, 1, 1],
            [3,   2,  192, 0, 1, 1],
            [3,   2,  192, 1, 1, 1],
            [3,   2, 192, 0, 1, 1],
            [3,   2, 192, 1, 1, 1],
            [3,   2, 192, 0, 1, 1],
            [3,   2, 192, 1, 1, 1],
            [3,   2, 192, 0, 1, 1],
            [3,   2, 192, 1, 1, 1],
            [3,   2, 192, 0, 1, 1],
            [3,   2, 192, 1, 1, 1],
            [3,   2, 192, 0, 1, 1],
            [3,   2, 192, 1, 1, 1],
            [3,   2, 192, 0, 1, 1],
            [3,   2, 192, 0, 1, 1],
            [3,   2, 384, 0, 1, 2],
            [3,   2, 384, 1, 1, 1],
            [3,   2, 384, 0, 1, 1]
        ],
        feat_out_indice = [7,23,26],
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='ckpt/new_repvit_m0_9_distill_450e.pth',
        #     prefix='backbone.',
        # )
        ),
    neck=dict(
        type='HybridEncoder',
        in_channels=[96,192,384],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1,2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2)),
    head=dict(
        type='RTMOHead',
        num_keypoints=21,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            cls_feat_channels=256,
            channels_per_group=36,
            pose_vec_channels=256,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(type='PoseOKS', metainfo=metafile),
            use_keypoints_for_center=True),
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]),
        dcc_cfg=dict(
            in_channels=256,
            feat_channels=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_cfg=dict(
                s=128,
                expansion_factor=2,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn='SiLU',
                pos_enc='add')),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo=metafile,
            loss_weight=30.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    ))
