from mmpose.datasets.transforms.common_transforms import YOLOXHSVRandomAug,RandomFlip,GenerateTarget
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
            if isinstance(results['left_img'], list):
                results['left_img'] = [
                    imflip(img, direction=flip_dir) for img in results['left_img']
                ]
            else:
                results['left_img'] = imflip(results['left_img'], direction=flip_dir)
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

        results['left_img'] = _transform(results['left_img'])
        results['right_img'] = _transform(results['right_img'])
        return results