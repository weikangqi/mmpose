from mmpose.datasets.transforms.common_transforms import YOLOXHSVRandomAug,RandomFlip,GenerateTarget,FilterAnnotations
import numpy as np
import cv2
from mmpose.registry import TRANSFORMS
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmcv.image import imflip
from mmpose.structures.bbox import bbox_xyxy2cs, flip_bbox
from mmpose.structures.keypoint import flip_keypoints
from mmpose.utils.typing import MultiConfig


@TRANSFORMS.register_module()
class StereoFilterAnnotations(FilterAnnotations):
    
    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_area: int = 1,
                 min_kpt_vis: int = 1,
                 by_box: bool = False,
                 by_area: bool = False,
                 by_kpt: bool = True,
                 keep_empty: bool = True) -> None:
        super().__init__(
            min_gt_bbox_wh,
            min_gt_area,
            min_kpt_vis,
            by_box,
            by_area,
            by_kpt,
            keep_empty
        )
        
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        import copy
        y = copy.deepcopy(results)
        assert 'keypoints' in results
        kpts = results['keypoints']
        if kpts.shape[0] == 0:
            return results

        tests = []
        tests_right = []
        if self.by_box and 'bbox' in results:
            bbox = results['bbox']
            tests.append(
                ((bbox[..., 2] - bbox[..., 0] > self.min_gt_bbox_wh[0]) &
                 (bbox[..., 3] - bbox[..., 1] > self.min_gt_bbox_wh[1])))
            
        if self.by_box and 'right_bbox' in results:
            bbox = results['right_bbox']
            tests_right.append(
                ((bbox[..., 2] - bbox[..., 0] > self.min_gt_bbox_wh[0]) &
                 (bbox[..., 3] - bbox[..., 1] > self.min_gt_bbox_wh[1])))
            
        if self.by_area and 'area' in results:
            area = results['area']
            tests.append(area >= self.min_gt_area)
        if self.by_area and 'right_area' in results:
            area_right = results['right_area']
            tests_right.append(area_right >= self.min_gt_area)
        if self.by_kpt:
            kpts_vis = results['keypoints_visible']
            if kpts_vis.ndim == 3:
                kpts_vis = kpts_vis[..., 0]
            tests.append(kpts_vis.sum(axis=1) >= self.min_kpt_vis)
        if self.by_kpt:
            kpts_vis = results['right_keypoints_visible']
            if kpts_vis.ndim == 3:
                kpts_vis = kpts_vis[..., 0]
            tests_right.append(kpts_vis.sum(axis=1) >= self.min_kpt_vis)

        keep = tests[0]
        keep_right = tests_right[0]
        for t in tests[1:]:
            keep = keep & t
        for t in tests_right[1:]:
            keep_right = keep_right & t

        if not keep.any() or keep_right.any():
            if self.keep_empty:
                return None

        # 修改bbox_score -> bbox_scores
        # if(type('bbox') )
        keys = ('bbox', 'bbox_score', 'category_id', 'keypoints',
                'keypoints_visible', 'area')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]
                
        keys = ('right_bbox', 'right_bbox_score', 'right_keypoints',
                'right_keypoints_visible','right_area')
        for key in keys:
            if key in results:
                results[key] = results[key][keep_right]    
        if not  (keep_right == keep).all():
            print("dd")
        # assert (keep_right == keep).all()
        # assert len(results['area']) ==  len(results['right_area'])

        # assert  len(results['keypoints']) == len(results['right_keypoints'])

        # assert len(results['keypoints_visible']) == len(results['right_keypoints_visible'])
            
        # assert len(results['bbox']) == len(results['right_bbox'])
        return results



@TRANSFORMS.register_module()
class StereoGenerateTarget(GenerateTarget):
    
    def __init__(self,
                 encoder: MultiConfig,
                 target_type: Optional[str] = None,
                 multilevel: bool = False,
                 use_dataset_keypoint_weights: bool = False) -> None:
        super().__init__(
            encoder,
            target_type,
            multilevel,
            use_dataset_keypoint_weights
        )
        
        
    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GenerateTarget`.

        See ``transform()`` method of :class:`BaseTransform` for details.
        """


        right_keypoints_visible = results['right_keypoints_visible']
        if right_keypoints_visible.ndim == 3 and right_keypoints_visible.shape[2] == 2:
            right_keypoints_visible, right_keypoints_visible_weights = \
                right_keypoints_visible[..., 0], right_keypoints_visible[..., 1]
            results['right_keypoints_visible'] = right_keypoints_visible
            results['right_keypoints_visible_weights'] = right_keypoints_visible_weights
        
        if results.get('transformed_keypoints', None) is not None:
            # use keypoints transformed by TopdownAffine
            keypoints = results['transformed_keypoints']
        elif results.get('keypoints', None) is not None:
            # use original keypoints
            keypoints = results['keypoints']
        else:
            raise ValueError(
                'GenerateTarget requires \'transformed_keypoints\' or'
                ' \'keypoints\' in the results.')

        keypoints_visible = results['keypoints_visible']
        if keypoints_visible.ndim == 3 and keypoints_visible.shape[2] == 2:
            keypoints_visible, keypoints_visible_weights = \
                keypoints_visible[..., 0], keypoints_visible[..., 1]
            results['keypoints_visible'] = keypoints_visible
            results['keypoints_visible_weights'] = keypoints_visible_weights
            
            
        

        # Encoded items from the encoder(s) will be updated into the results.
        # Please refer to the document of the specific codec for details about
        # encoded items.
        if not isinstance(self.encoder, list):
            # For single encoding, the encoded items will be directly added
            # into results.
            auxiliary_encode_kwargs = {
                key: results.get(key, None)
                for key in self.encoder.auxiliary_encode_keys
            }
            encoded = self.encoder.encode(
                keypoints=keypoints,
                keypoints_visible=keypoints_visible,
                **auxiliary_encode_kwargs)

            if self.encoder.field_mapping_table:
                encoded[
                    'field_mapping_table'] = self.encoder.field_mapping_table
            if self.encoder.instance_mapping_table:
                encoded['instance_mapping_table'] = \
                    self.encoder.instance_mapping_table
            if self.encoder.label_mapping_table:
                encoded[
                    'label_mapping_table'] = self.encoder.label_mapping_table

        else:
            encoded_list = []
            _field_mapping_table = dict()
            _instance_mapping_table = dict()
            _label_mapping_table = dict()
            for _encoder in self.encoder:
                auxiliary_encode_kwargs = {
                    key: results[key]
                    for key in _encoder.auxiliary_encode_keys
                }
                encoded_list.append(
                    _encoder.encode(
                        keypoints=keypoints,
                        keypoints_visible=keypoints_visible,
                        **auxiliary_encode_kwargs))

                _field_mapping_table.update(_encoder.field_mapping_table)
                _instance_mapping_table.update(_encoder.instance_mapping_table)
                _label_mapping_table.update(_encoder.label_mapping_table)

            if self.multilevel:
                # For multilevel encoding, the encoded items from each encoder
                # should have the same keys.

                keys = encoded_list[0].keys()
                if not all(_encoded.keys() == keys
                           for _encoded in encoded_list):
                    raise ValueError(
                        'Encoded items from all encoders must have the same '
                        'keys if ``multilevel==True``.')

                encoded = {
                    k: [_encoded[k] for _encoded in encoded_list]
                    for k in keys
                }

            else:
                # For combined encoding, the encoded items from different
                # encoders should have no overlapping items, except for
                # `keypoint_weights`. If multiple `keypoint_weights` are given,
                # they will be multiplied as the final `keypoint_weights`.

                encoded = dict()
                keypoint_weights = []

                for _encoded in encoded_list:
                    for key, value in _encoded.items():
                        if key == 'keypoint_weights':
                            keypoint_weights.append(value)
                        elif key not in encoded:
                            encoded[key] = value
                        else:
                            raise ValueError(
                                f'Overlapping item "{key}" from multiple '
                                'encoders, which is not supported when '
                                '``multilevel==False``')

                if keypoint_weights:
                    encoded['keypoint_weights'] = keypoint_weights

            if _field_mapping_table:
                encoded['field_mapping_table'] = _field_mapping_table
            if _instance_mapping_table:
                encoded['instance_mapping_table'] = _instance_mapping_table
            if _label_mapping_table:
                encoded['label_mapping_table'] = _label_mapping_table

        if self.use_dataset_keypoint_weights and 'keypoint_weights' in encoded:
            if isinstance(encoded['keypoint_weights'], list):
                for w in encoded['keypoint_weights']:
                    w = w * results['dataset_keypoint_weights']
            else:
                encoded['keypoint_weights'] = encoded[
                    'keypoint_weights'] * results['dataset_keypoint_weights']

        
        results.update(encoded)

        return results
    

@TRANSFORMS.register_module()
class StereoRandomFlip(RandomFlip):

    def __init__(self, prob: Union[float, List[float]] = 0.5,
                 direction: Union[str, List[str]] = 'horizontal') -> None:
        super().__init__(prob, direction)


    def transform(self, results: dict) -> dict:
        """The transform function of :class:`RandomFlip`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        flip_dir = self._choose_direction()

        if flip_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = flip_dir

            h, w = results.get('input_size', results['img_shape'])
            # flip image and mask
            if isinstance(results['img'], list):
                results['img'] = [
                    imflip(img, direction=flip_dir) for img in results['img']
                ]
            else:
                results['img'] = imflip(results['img'], direction=flip_dir)
            if isinstance(results['right_img'], list):
                results['right_img'] = [
                    imflip(img, direction=flip_dir) for img in results['right_img']
                ]
            else:
                results['right_img'] = imflip(results['right_img'], direction=flip_dir)    
            

            if 'img_mask' in results:
                results['img_mask'] = imflip(
                    results['img_mask'], direction=flip_dir)

            # flip bboxes
            if results.get('bbox', None) is not None:
                results['bbox'] = flip_bbox(
                    results['bbox'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)
            if results.get('right_bbox', None) is not None:
                results['right_bbox'] = flip_bbox(
                    results['right_bbox'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)

            if results.get('bbox_center', None) is not None:
                results['bbox_center'] = flip_bbox(
                    results['bbox_center'],
                    image_size=(w, h),
                    bbox_format='center',
                    direction=flip_dir)

            # flip keypoints
            if results.get('keypoints', None) is not None:
                keypoints, keypoints_visible = flip_keypoints(
                    results['keypoints'],
                    results.get('keypoints_visible', None),
                    image_size=(w, h),
                    flip_indices=results['flip_indices'],
                    direction=flip_dir)

                results['keypoints'] = keypoints
                results['keypoints_visible'] = keypoints_visible
            if results.get('right_keypoints', None) is not None:
                keypoints, keypoints_visible = flip_keypoints(
                    results['right_keypoints'],
                    results.get('right_keypoints_visible', None),
                    image_size=(w, h),
                    flip_indices=results['flip_indices'],
                    direction=flip_dir)

                results['right_keypoints'] = keypoints
                results['right_keypoints_visible'] = keypoints_visible

        return results

@TRANSFORMS.register_module()
class StereoYOLOXHSVRandomAug(YOLOXHSVRandomAug):
    def __init__(self, hue_delta: int = 5, saturation_delta: int = 30, value_delta: int = 30) -> None:
        super().__init__(hue_delta, saturation_delta, value_delta)
        


    def transform(self, results: dict) -> dict:
        def _transform(img):
            hsv_gains = self._get_hsv_gains()
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
            cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
            return img

        results['img'] = _transform(results['img'])
        if 'right_img' in results:
            results['right_img'] = _transform(results['right_img'])
        return results