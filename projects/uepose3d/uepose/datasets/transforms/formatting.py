from mmpose.datasets.transforms import PackPoseInputs
from mmpose.datasets.transforms.formatting import image_to_tensor,keypoints_to_tensor
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from mmpose.codecs.base import BaseKeypointCodec
from mmpose.codecs.utils.refinement import refine_simcc_dark
# from mmpose.registry import KEYPOINT_CODECS
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures import MultilevelPixelData, PoseDataSample


@TRANSFORMS.register_module()
class StereoPackPoseInputs(PackPoseInputs):
    
    
    def __init__(self, meta_keys=..., pack_transformed=False):
        super().__init__(meta_keys, pack_transformed)
        
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`PoseDataSample`): The annotation info of the
                sample.
        """
        # Pack image(s) for 2d pose estimation
        if 'img' in results:
            img = results['img']
            inputs_tensor = image_to_tensor(img)
        if  'right_img' in results:
            right_img = results['right_img']
            inputs_tensor = torch.cat([inputs_tensor,image_to_tensor(right_img)],dim=-1)
        # Pack keypoints for 3d pose-lifting
        elif 'lifting_target' in results and 'keypoints' in results:
            if 'keypoint_labels' in results:
                keypoints = results['keypoint_labels']
            else:
                keypoints = results['keypoints']
            inputs_tensor = keypoints_to_tensor(keypoints)

        data_sample3d = PoseDataSample()

        # pack instance data
        gt_instances_3d = InstanceData()
        _instance_mapping_table = results.get('instance_mapping_table',
                                              self.instance_mapping_table)
        for key, packed_key in _instance_mapping_table.items():
            if key in results:
                gt_instances_3d.set_field(results[key], packed_key)

        # pack `transformed_keypoints` for visualizing data transform
        # and augmentation results
        if self.pack_transformed and 'transformed_keypoints' in results:
            gt_instances_3d.set_field(results['transformed_keypoints'],
                                   'transformed_keypoints')

        data_sample3d.gt_instances = gt_instances_3d

        # pack instance labels
        gt_instance_labels = InstanceData()
        _label_mapping_table = results.get('label_mapping_table',
                                           self.label_mapping_table)
        for key, packed_key in _label_mapping_table.items():
            if key in results:
                if isinstance(results[key], list):
                    _labels = np.stack(results[key])
                    gt_instance_labels.set_field(_labels, packed_key)
                else:
                    gt_instance_labels.set_field(results[key], packed_key)
        data_sample3d.gt_instance_labels = gt_instance_labels.to_tensor()

        # pack fields
        gt_fields = None
        _field_mapping_table = results.get('field_mapping_table',
                                           self.field_mapping_table)
        for key, packed_key in _field_mapping_table.items():
            if key in results:
                if isinstance(results[key], list):
                    if gt_fields is None:
                        gt_fields = MultilevelPixelData()
                    else:
                        assert isinstance(
                            gt_fields, MultilevelPixelData
                        ), 'Got mixed single-level and multi-level pixel data.'
                else:
                    if gt_fields is None:
                        gt_fields = PixelData()
                    else:
                        assert isinstance(
                            gt_fields, PixelData
                        ), 'Got mixed single-level and multi-level pixel data.'

                gt_fields.set_field(results[key], packed_key)

        if gt_fields:
            data_sample3d.gt_fields = gt_fields.to_tensor()

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample3d.set_metainfo(img_meta)

        packed_results = dict()
        packed_results['inputs'] = inputs_tensor
        packed_results['data_samples'] = data_sample3d

        return packed_results
        
        
    