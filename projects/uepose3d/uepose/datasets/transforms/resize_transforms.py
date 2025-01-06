from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import xtcocotools.mask as cocomask
from mmcv.image import imflip_, imresize
from mmcv.image.geometric import imrescale
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from scipy.stats import truncnorm

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import (bbox_clip_border, bbox_corner2xyxy,
                                    bbox_xyxy2corner, get_pers_warp_matrix,
                                    get_udp_warp_matrix, get_warp_matrix)
from mmpose.structures.keypoint import keypoint_clip_border
@TRANSFORMS.register_module()
class StereoInputResize(BaseTransform):
    """Resize the image to the input size of the model. Optionally, the image
    can be resized to multiple sizes to build a image pyramid for multi-scale
    inference.

    Required Keys:

        - img
        - ori_shape

    Modified Keys:

        - img
        - img_shape

    Added Keys:

        - input_size
        - warp_mat
        - aug_scale

    Args:
        input_size (Tuple[int, int]): The input size of the model in [w, h].
            Note that the actually size of the resized image will be affected
            by ``resize_mode`` and ``size_factor``, thus may not exactly equals
            to the ``input_size``
        aug_scales (List[float], optional): The extra input scales for
            multi-scale testing. If given, the input image will be resized
            to different scales to build a image pyramid. And heatmaps from
            all scales will be aggregated to make final prediction. Defaults
            to ``None``
        size_factor (int): The actual input size will be ceiled to
                a multiple of the `size_factor` value at both sides.
                Defaults to 16
        resize_mode (str): The method to resize the image to the input size.
            Options are:

                - ``'fit'``: The image will be resized according to the
                    relatively longer side with the aspect ratio kept. The
                    resized image will entirely fits into the range of the
                    input size
                - ``'expand'``: The image will be resized according to the
                    relatively shorter side with the aspect ratio kept. The
                    resized image will exceed the given input size at the
                    longer side
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,input_size: Tuple[int, int]):
        super().__init__()
        self.input_size = input_size

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupResize` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        img_h, img_w = results['ori_shape']
        w, h = self.input_size

        input_size = [(w, h)]
        # if self.aug_scales:
        #     input_sizes += [(int(w * s), int(h * s)) for s in self.aug_scales]

        padded_input_size = [w,h]
        actual_input_size = [img_w, img_h]

        # actual_input_size, padded_input_size = self._get_input_size(
        #         img_size=(img_w, img_h), input_size=(_w, _h))

            # if self.use_udp:
            #     center = np.array([(img_w - 1.0) / 2, (img_h - 1.0) / 2],
            #                       dtype=np.float32)
            #     scale = np.array([img_w, img_h], dtype=np.float32)
            #     warp_mat = get_udp_warp_matrix(
            #         center=center,
            #         scale=scale,
            #         rot=0,
            #         output_size=actual_input_size)
            # else:
        center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
        scale = np.array([
                    img_w * padded_input_size[0] / actual_input_size[0],
                    img_h * padded_input_size[1] / actual_input_size[1]
                ],
                                 dtype=np.float32)


            # Store the transform information w.r.t. the main input size
            # if i == 0:
        results['img_shape'] = padded_input_size[::-1]
        results['input_center'] = center
        results['input_scale'] = scale
        results['input_size'] = padded_input_size

        # if self.aug_scales:
        #     results['img'] = imgs
        #     results['aug_scales'] = self.aug_scales
        # else:
        #     results['img'] = imgs[0]
        #     results['aug_scale'] = None

        return results
