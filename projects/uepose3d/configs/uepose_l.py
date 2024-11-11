_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(imports=['uepose'], allow_failed_imports=False)
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='StereoPose3dLocalVisualizerPlus', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=600, val_interval=20, dynamic_intervals=[(580, 1)])


# data
input_size = (640, 640)
metafile = 'configs/_base_/datasets/uepose.py'
# codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)
codec = dict(
    type='StereoSimCC3DLabel',
    input_size=(640, 640, 640),
    sigma=(6., 6.93, 6.),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    root_index=(11, 12))

train_pipeline = [
    dict(type='LoadStereoImage'),
    dict(
        type='StereoBottomupRandomAffine',
        input_size=input_size,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='StereoYOLOXHSVRandomAug'),
    dict(type='StereoRandomFlip',direction='horizontal'),
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
data_root = '/workspace/MobileHumanPose3D/datasets'

# train datasets
dataset_coco = dict(
    type='UnrealPose3dDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='uecoco_3d/annotations/test_stereo.json',
    data_prefix=dict(img='uecoco_3d/test_stereo'),
    pipeline=train_pipeline,
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_coco)

val_pipeline = [
    dict(type='LoadStereoImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='UnrealPose3dDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='uecoco_3d/annotations/test_stereo.json',
        data_prefix=dict(img='uecoco_3d/test_stereo'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader




