# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.fileio import exists, get_local_path
from mmengine.utils import is_abs

from ._base.base_stereo_view_dataset import BaseStereoViewDataset
from mmpose.registry import DATASETS
from xtcocotools.coco import COCO

@DATASETS.register_module()
class UnrealPose3dDataset(BaseStereoViewDataset):
    """
    Args:
        ann_file (str): Annotation file path. Default: ''.
        seq_len (int): Number of frames in a sequence. Default: 1.
        seq_step (int): The interval for extracting frames from the video.
            Default: 1.
        multiple_target (int): If larger than 0, merge every
            ``multiple_target`` sequence together. Default: 0.
        multiple_target_step (int): The interval for merging sequence. Only
            valid when ``multiple_target`` is larger than 0. Default: 0.
        pad_video_seq (bool): Whether to pad the video so that poses will be
            predicted for every frame in the video. Default: ``False``.
        causal (bool): If set to ``True``, the rightmost input frame will be
            the target frame. Otherwise, the middle input frame will be the
            target frame. Default: ``True``.
        subset_frac (float): The fraction to reduce dataset size. If set to 1,
            the dataset size is not reduced. Default: 1.
        keypoint_2d_src (str): Specifies 2D keypoint information options, which
            should be one of the following options:

            - ``'gt'``: load from the annotation file
            - ``'detection'``: load from a detection
              result file of 2D keypoint
            - 'pipeline': the information will be generated by the pipeline

            Default: ``'gt'``.
        keypoint_2d_det_file (str, optional): The 2D keypoint detection file.
            If set, 2d keypoint loaded from this file will be used instead of
            ground-truth keypoints. This setting is only when
            ``keypoint_2d_src`` is ``'detection'``. Default: ``None``.
        factor_file (str, optional): The projection factors' file. If set,
            factor loaded from this file will be used instead of calculated
            factors. Default: ``None``.
        camera_param_file (str): Cameras' parameters file. Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/uepose.py')
    SUPPORTED_keypoint_2d_src = {'gt', 'detection', 'pipeline'}

    def __init__(self,
                 ann_file: str = '',
                 seq_len: int = 1,
                 multiple_target: int = 0,
                 causal: bool = True,
                 subset_frac: float = 1.0,
                 camera_param_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
 

        super().__init__(
            ann_file=ann_file,
            seq_len=seq_len,
            multiple_target=multiple_target,
            causal=causal,
            subset_frac=subset_frac,
            camera_param_file=camera_param_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def _load_ann_file(self, ann_file: str) -> dict:
        """Load annotation file."""
        with get_local_path(ann_file) as local_path:
            self.ann_data = COCO(local_path)
        # pass


    def get_sequence_indices(self) -> List[List[int]]:
        """Build sequence indices.

        The default method creates sample indices that each sample is a single
        frame (i.e. seq_len=1). Override this method in the subclass to define
        how frames are sampled to form data samples.

        Outputs:
            sample_indices: the frame indices of each sample.
                For a sample, all frames will be treated as an input sequence,
                and the ground-truth pose of the last frame will be the target.
        """
        sequence_indices = []
        if self.seq_len == 1:
            num_imgs = len(self.ann_data.anns)
            sequence_indices = [[idx+1] for idx in range(num_imgs)]
        else:
            raise NotImplementedError('Multi-frame data sample unsupported!')

        return sequence_indices
    
    
    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        num_keypoints = 21
        self._metainfo['CLASSES'] = self.ann_data.loadCats(
            self.ann_data.getCatIds())
        instance_list = []
        image_list = []
        for i, _ann_ids in enumerate(self.sequence_indices):
            anns = self.ann_data.loadAnns(_ann_ids)
            num_anns = len(anns)
            img_ids = []
            kpts = np.zeros((num_anns, num_keypoints, 2), dtype=np.float32)
            right_kpts = np.zeros((num_anns, num_keypoints, 2), dtype=np.float32)
            kpts_3d = np.zeros((num_anns, num_keypoints, 3), dtype=np.float32)
            keypoints_visible = np.zeros((num_anns, num_keypoints),
                                            dtype=np.float32)
            right_keypoints_visible = np.zeros((num_anns, num_keypoints),
                                            dtype=np.float32)
            scales = np.zeros((num_anns, 2), dtype=np.float32)
            centers = np.zeros((num_anns, 2), dtype=np.float32)
            bboxes = np.zeros((num_anns, 4), dtype=np.float32)
            bbox_scores = np.zeros((num_anns, 1), dtype=np.float32)
            bbox_scales = np.zeros((num_anns, 2), dtype=np.float32)
            right_bboxes = np.zeros((num_anns, 4), dtype=np.float32)
            right_bbox_scores = np.zeros((num_anns, 1), dtype=np.float32)
            right_bbox_scales = np.zeros((num_anns, 2), dtype=np.float32)

            for j, ann in enumerate(anns):
                img_ids.append(ann['image_id'])
                kpts[j] = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)[:, :2]
                
                
                kpts_3d[j] = np.array(ann['keypoints_3d'], dtype=np.float32).reshape(-1,4)[:, :3]
                keypoints_visible[j] = np.array(
                    ann['keypoints'], dtype=np.float32).reshape(-1,3)[:, 2]
                keypoints_visible[j] = np.minimum(1, keypoints_visible[j])
                
                if 'right_keypoints' in ann:
                    right_kpts[j] = np.array(ann['right_keypoints'], dtype=np.float32).reshape(-1,3)[:, :2]
                    right_keypoints_visible[j] = np.array(ann['right_keypoints'], dtype=np.float32).reshape(-1,3)[:, 2]
                    right_keypoints_visible[j] = np.minimum(1, right_keypoints_visible[j])
                if 'scale' in ann:
                    scales[j] = np.array(ann['scale'])
                if 'center' in ann:
                    centers[j] = np.array(ann['center'])
                bboxes[j] = np.array(ann['bbox'], dtype=np.float32)
                bbox_scores[j] = np.array([1], dtype=np.float32)
                bbox_scales[j] = np.array([1, 1], dtype=np.float32)
                
                if 'right_bbox' in ann:
                    right_bboxes[j] = np.array(ann['right_bbox'], dtype=np.float32)
                    right_bbox_scores[j] = np.array([1], dtype=np.float32)
                    right_bbox_scales[j] = np.array([1, 1], dtype=np.float32)

            imgs = self.ann_data.loadImgs(img_ids)

            img_paths = np.array(
                 [
                [f'{self.data_prefix["img"]}/' + img['left_file_name'],f'{self.data_prefix["img"]}/' + img['right_file_name'] ] for img in imgs
            ]
            )
            factors = np.zeros((kpts_3d.shape[0], ), dtype=np.float32)

            target_idx = [-1] if self.causal else [int(self.seq_len // 2)]
            if self.multiple_target:
                target_idx = list(range(self.multiple_target))

            cam_param = anns[-1]['camera_param']
   
            cam_param['w'] = 640
            cam_param['h'] = 480
            cam_param = {'f': [cam_param['K'][0][0],cam_param['K'][1][1]], 'c': [cam_param['K'][0][2],cam_param['K'][1][2]]}
            x,y,max_x,max_y = bboxes[0]
            x,y,max_x,max_y = bboxes[0]
            instance_info = {
                'num_keypoints': num_keypoints,
                'keypoints': kpts,
                'keypoints_3d': kpts_3d,
                'keypoints_visible': keypoints_visible,
                'scale': scales,
                'center': centers,
                'id': i,
                'category_id': 1,
                'iscrowd': 0,
                'img_path': img_paths,
                'img_ids': [img['id'] for img in imgs],
                'img_id': imgs[0]['id'],
                'lifting_target': kpts_3d[target_idx],
                'lifting_target_visible': keypoints_visible[target_idx],
                'target_img_paths': list(img_paths[target_idx]),
                'camera_param': [cam_param],
                'factor': factors,
                'target_idx': target_idx,
                'bbox': np.array([[bboxes[0][0],bboxes[0][1],bboxes[0][0]+bboxes[0][2],bboxes[0][1]+bboxes[0][3]]]),
                # 'bbox_scales': bbox_scales,
                'bbox_score': bbox_scores,
                'area': np.clip((bboxes[0][2] * bboxes[0][3])*0.53,a_min=1.0,a_max=None),# [x_min, y_min, x_max, y_max]
                'right_keypoints': right_kpts,
                'right_keypoints_visible': right_keypoints_visible,
                'right_bbox': np.array([[right_bboxes[0][0],right_bboxes[0][1],right_bboxes[0][0]+right_bboxes[0][2],right_bboxes[0][1]+right_bboxes[0][3]]]),
                # 'right_bbox_scales': right_bbox_scales,
                'right_bbox_score': right_bbox_scores,
                'right_area':np.clip((right_bboxes[0][2] * right_bboxes[0][3])*0.53,a_min=1.0,a_max=None)
            }

            instance_list.append(instance_info)

        return instance_list, image_list

