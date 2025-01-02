from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
from mmpose.datasets.transforms import BottomupRandomAffine,BottomupResize
from mmpose.structures.bbox import (bbox_clip_border, bbox_corner2xyxy,
                                    bbox_xyxy2corner, get_pers_warp_matrix,
                                    get_udp_warp_matrix, get_warp_matrix)
from mmpose.structures.keypoint import keypoint_clip_border

import cv2
import numpy as np


@TRANSFORMS.register_module()
class StereoBottomupResize(BottomupResize):
    def __init__(self,
                 input_size: Tuple[int, int],
                 aug_scales: Optional[List[float]] = None,
                 size_factor: int = 32,
                 resize_mode: str = 'fit',
                 pad_val: tuple = (0, 0, 0),
                 use_udp: bool = False):
        super().__init__(
            input_size,
            aug_scales,
            size_factor,
            resize_mode,
            pad_val,
            use_udp
        )

    def transform(self, results: Dict) -> Optional[dict]:
        results = self._transform(results,'img')
        results = self._transform(results,'right_img')
        return results

    def _transform(self, results: Dict, img_keys: str) -> Optional[dict]:
        """The transform function of :class:`BottomupResize` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results[img_keys]
        img_h, img_w = results['ori_shape']
        w, h = self.input_size

        input_sizes = [(w, h)]
        if self.aug_scales:
            input_sizes += [(int(w * s), int(h * s)) for s in self.aug_scales]

        imgs = []
        for i, (_w, _h) in enumerate(input_sizes):

            actual_input_size, padded_input_size = self._get_input_size(
                img_size=(img_w, img_h), input_size=(_w, _h))

            if self.use_udp:
                center = np.array([(img_w - 1.0) / 2, (img_h - 1.0) / 2],
                                  dtype=np.float32)
                scale = np.array([img_w, img_h], dtype=np.float32)
                warp_mat = get_udp_warp_matrix(
                    center=center,
                    scale=scale,
                    rot=0,
                    output_size=actual_input_size)
            else:
                center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
                scale = np.array([
                    img_w * padded_input_size[0] / actual_input_size[0],
                    img_h * padded_input_size[1] / actual_input_size[1]
                ],
                                 dtype=np.float32)
                warp_mat = get_warp_matrix(
                    center=center,
                    scale=scale,
                    rot=0,
                    output_size=padded_input_size)

            _img = cv2.warpAffine(
                img,
                warp_mat,
                padded_input_size,
                flags=cv2.INTER_LINEAR,
                borderValue=self.pad_val)

            imgs.append(_img)

            # Store the transform information w.r.t. the main input size
            if i == 0:
                results['img_shape'] = padded_input_size[::-1]
                results['input_center'] = center
                results['input_scale'] = scale
                results['input_size'] = padded_input_size

        if self.aug_scales:
            results[img_keys] = imgs
            results['aug_scales'] = self.aug_scales
        else:
            results[img_keys] = imgs[0]
            results['aug_scale'] = None

        return results


@TRANSFORMS.register_module()
class StereoBottomupRandomAffine(BottomupRandomAffine):
    def __init__(self,
                 input_size: Optional[Tuple[int, int]] = None,
                 shift_factor: float = 0.2,
                 shift_prob: float = 1.,
                 scale_factor: Tuple[float, float] = (0.75, 1.5),
                 scale_prob: float = 1.,
                 scale_type: str = 'short',
                 rotate_factor: float = 30.,
                 rotate_prob: float = 1,
                 shear_factor: float = 2.0,
                 shear_prob: float = 1.0,
                 use_udp: bool = False,
                 pad_val: Union[float, Tuple[float]] = 0,
                 border: Tuple[int, int] = (0, 0),
                 distribution='trunc_norm',
                 transform_mode='affine',
                 bbox_keep_corner: bool = True,
                 clip_border: bool = False) -> None:
        super().__init__(
            input_size,
            shift_factor,
            shift_prob,
            scale_factor,
            scale_prob,
            scale_type,
            rotate_factor,
            rotate_prob,
            shear_factor,
            shear_prob,
            use_udp,
            pad_val,
            border,
            distribution,
            transform_mode,
            bbox_keep_corner,
            clip_border
        )
        
        
    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupRandomAffine` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        img_h, img_w = results['img_shape'][:2]
        w, h = self.input_size

        offset_rate, scale_rate, rotate, shear = self._get_transform_params()

        if 'affine' in self.transform_mode:
            offset = offset_rate * [img_w, img_h]
            scale = scale_rate * [img_w, img_h]
            # adjust the scale to match the target aspect ratio
            scale = self._fix_aspect_ratio(scale, aspect_ratio=w / h)

            if self.transform_mode == 'affine_udp':
                center = np.array([(img_w - 1.0) / 2, (img_h - 1.0) / 2],
                                  dtype=np.float32)
                warp_mat = get_udp_warp_matrix(
                    center=center + offset,
                    scale=scale,
                    rot=rotate,
                    output_size=(w, h))
            else:
                center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
                warp_mat = get_warp_matrix(
                    center=center + offset,
                    scale=scale,
                    rot=rotate,
                    output_size=(w, h))

        else:
            offset = offset_rate * [w, h]
            center = np.array([w / 2, h / 2], dtype=np.float32)
            warp_mat = get_pers_warp_matrix(
                center=center,
                translate=offset,
                scale=scale_rate[0],
                rot=rotate,
                shear=shear)
            
        # warp image and keypoints
        results['img'] = self._transform(results['img'], warp_mat,
                                         (int(w), int(h)))
        
        if 'right_img' in results:
            results['right_img'] = self._transform(results['right_img'], warp_mat,
                                            (int(w), int(h)))

        if 'keypoints' in results:
            # Only transform (x, y) coordinates
            kpts = cv2.transform(results['keypoints'], warp_mat)
            if kpts.shape[-1] == 3:
                kpts = kpts[..., :2] / kpts[..., 2:3]
            results['keypoints'] = kpts

            if self.clip_border:
                results['keypoints'], results[
                    'keypoints_visible'] = keypoint_clip_border(
                        results['keypoints'], results['keypoints_visible'],
                        (w, h))

        if 'right_keypoints' in results:
            # Only transform (x, y) coordinates
            kpts = cv2.transform(results['right_keypoints'], warp_mat)
            if kpts.shape[-1] == 3:
                kpts = kpts[..., :2] / kpts[..., 2:3]
            results['right_keypoints'] = kpts

            if self.clip_border:
                results['right_keypoints'], results[
                    'right_keypoints_visible'] = keypoint_clip_border(
                        results['right_keypoints'], results['right_keypoints_visible'],
                        (w, h))
        else:
            results['right_keypoints'] =  results['keypoints']
            results['right_keypoints_visible'] = results['keypoints_visible']


        if 'bbox' in results:
            bbox = bbox_xyxy2corner(results['bbox'])
            bbox = cv2.transform(bbox, warp_mat)
            if bbox.shape[-1] == 3:
                bbox = bbox[..., :2] / bbox[..., 2:3]
            if not self.bbox_keep_corner:
                bbox = bbox_corner2xyxy(bbox)
            if self.clip_border:
                bbox = bbox_clip_border(bbox, (w, h))
            results['bbox'] = bbox
            
        if 'right_bbox' in results:
            bbox = bbox_xyxy2corner(results['right_bbox'])
            bbox = cv2.transform(bbox, warp_mat)
            if bbox.shape[-1] == 3:
                bbox = bbox[..., :2] / bbox[..., 2:3]
            if not self.bbox_keep_corner:
                bbox = bbox_corner2xyxy(bbox)
            if self.clip_border:
                bbox = bbox_clip_border(bbox, (w, h))
            results['right_bbox'] = bbox
        else:
            results['right_bbox'] = results['bbox']

        if 'area' in results:
            warp_mat_for_area = warp_mat
            if warp_mat.shape[0] == 2:
                aux_row = np.array([[0.0, 0.0, 1.0]], dtype=warp_mat.dtype)
                warp_mat_for_area = np.concatenate((warp_mat, aux_row))
            results['area'] *= np.linalg.det(warp_mat_for_area)

        results['input_size'] = self.input_size
        results['warp_mat'] = warp_mat
        

        return results