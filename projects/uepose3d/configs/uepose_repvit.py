_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(imports=['uepose'], allow_failed_imports=False)

# runtime
train_cfg = dict(max_epochs=1000, val_interval=10, dynamic_intervals=[(580, 1)])

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))


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


# vis_backends = [
#     dict(type='LocalVisBackend'),
# ]
# visualizer = dict(
#     type='StereoPose3dLocalVisualizerPlus', vis_backends=vis_backends, name='visualizer')

# data
input_size = (640, 640)
metafile = 'configs/_base_/datasets/uepose.py'
codec = dict(type='StereoYOLOXPoseAnnotationProcessor', input_size=input_size)
# codec = dict(
#     type='StereoSimCC3DLabel',
#     input_size=(640, 640, 640),
#     sigma=(6., 6.93, 6.),
#     simcc_split_ratio=2.0,
#     normalize=False,
#     use_dark=False,
#     root_index=(11, 12))

train_pipeline = [
    dict(type='LoadStereoImage'),
    dict(
        type='StereoBottomupRandomAffine',
        input_size=input_size,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
        transform_mode='perspective',
    ),
    dict(type='StereoYOLOXHSVRandomAug'),
    dict(type='StereoRandomFlip',direction='horizontal'),
    dict(type='StereoFilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='StereoGenerateTarget', encoder=codec),
    dict(type='StereoPackPoseInputs', 
         meta_keys=('id',
                    'img_paths',
                    'keypoints',
                    'keypoints_visible',
                    'right_keypoints',
                    'right_keypoints_visible')),
]

data_mode = 'bottomup'
data_root = '/mmpose/data'
samll_preifix = ''
# train datasets
dataset_usepose = dict(
    type='UnrealPose3dDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='uecoco_3d/annotations/test_stereo.json',
    data_prefix=dict(img='uecoco_3d/test_stereo'),
    # pipeline=[
    #     dict(
    #         type='KeypointConverter',
    #         num_keypoints=21,
    #         mapping=[(i,i) for i in range(21)])
    # ]
)



# dataset_coco_train = dict(
#     type='CocoDataset',
#     data_root='data/',
#     data_mode=data_mode,
#     ann_file=f'coco/annotations/{samll_preifix}person_keypoints_train2017.json',
#     data_prefix=dict(img='coco/train2017'),
#     pipeline=[
#         dict(
#             type='KeypointConverter',
#             num_keypoints=21,
#             mapping=[(i,i) for i in range(17)])
#     ],
# )


train_dataset = dict(
    type='CombinedDataset',
    datasets=[dataset_usepose],
    pipeline=train_pipeline,
    metainfo=dict(from_file='/mmpose3d/configs/_base_/datasets/uepose.py'),
    test_mode=False)



train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)


val_dataset = train_dataset
val_dataloader = train_dataloader

val_pipeline = [
    dict(type='LoadStereoImage'),
    dict(
        type='StereoBottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(type='StereoPackPoseInputs', 
         meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]


dataset_coco_val = dict(
    type='CocoDataset',
    data_root='data/',
    data_mode=data_mode,
    ann_file=f'coco/annotations/{samll_preifix}person_keypoints_val2017.json',
    data_prefix=dict(img='coco/val2017'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=21,
            mapping=[(i,i) for i in range(17)])
    ],
)


val_dataset = dict(
    type='CombinedDataset',
    datasets=[dataset_usepose],
    pipeline=val_pipeline,
    metainfo=dict(from_file='/mmpose3d/configs/_base_/datasets/uepose.py'),
    test_mode=False)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset= val_dataset
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'/mmpose/data/uecoco_3d/annotations/test_stereo.json',
    score_mode='bbox',
    nms_mode='none',
    gt_converter= dict(
        type='KeypointConverter',
        num_keypoints=21,
        mapping=[(i, i) for i in range(17)])
)


test_evaluator = val_evaluator

# model
widen_factor = 0.5
deepen_factor = 0.33

model = dict(
    type='StereoBottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='StereoPoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
        ]),
    backbone=dict(
        type='RepViT',
        cfgs =   [
            # k, t, c, SE, HS, s 
            [3,   2,  80, 1, 0, 1],
            [3,   2,  80, 0, 0, 1],
            [3,   2,  80, 1, 0, 1],
            [3,   2,  80, 0, 0, 1],
            [3,   2,  80, 1, 0, 1],
            [3,   2,  80, 0, 0, 1],
            [3,   2,  80, 0, 0, 1],
            [3,   2,  160, 0, 0, 2],
            [3,   2,  160, 1, 0, 1],
            [3,   2,  160, 0, 0, 1],
            [3,   2,  160, 1, 0, 1],
            [3,   2,  160, 0, 0, 1],
            [3,   2,  160, 1, 0, 1],
            [3,   2,  160, 0, 0, 1],
            [3,   2,  160, 0, 0, 1],
            [3,   2,  320, 0, 1, 2],
            [3,   2,  320, 1, 1, 1],
            [3,   2,  320, 0, 1, 1],
            [3,   2,  320, 1, 1, 1],
            [3,   2,  320, 0, 1, 1],
            [3,   2,  320, 1, 1, 1],
            [3,   2,  320, 0, 1, 1],
            [3,   2,  320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 1, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 320, 0, 1, 1],
            [3,   2, 640, 0, 1, 2],
            [3,   2, 640, 1, 1, 1],
            [3,   2, 640, 0, 1, 1],
    ] ,
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/workspace/mmpose3d/ckpt/new_repvit_m2_3_distill_450e.pth',
        #     prefix='backbone.',
        # )
        ),
    neck=dict(
        type='HybridEncoder',
        in_channels=[160, 320, 640],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1, 2],
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


