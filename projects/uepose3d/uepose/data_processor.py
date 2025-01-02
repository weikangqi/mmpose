from mmpose.models.data_preprocessors import PoseDataPreprocessor
from typing import List, Optional, Sequence, Union
import math
from mmpose.registry import MODELS
import numpy as np
import torch
import torch.nn as nn
from mmengine.model import ImgDataPreprocessor
from mmengine.utils import is_seq_of
from mmengine.model.utils import stack_batch
import torch.nn.functional as F


@MODELS.register_module()
class StereoPoseDataPreprocessor(PoseDataPreprocessor):
    
    
    def __init__(self, mean: Sequence[float] = None, std: Sequence[float] = None, pad_size_divisor: int = 1, pad_value: float | int = 0, bgr_to_rgb: bool = False, rgb_to_bgr: bool = False, non_blocking: bool | None = False, batch_augments: List[dict] | None = None):
        super().__init__(mean, 
                         std, 
                         pad_size_divisor, 
                         pad_value, 
                         bgr_to_rgb, 
                         rgb_to_bgr, 
                         non_blocking, 
                         batch_augments)
        
        
    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        
        batch_pad_shape = self._get_pad_shape(data)
        data = self.cast_data(data)  # type: ignore
        data = self._forward(data=data, training=training,inputs_key='inputs')
        data = self._forward(data=data, training=training,inputs_key='right_inputs')
        left_inputs,right_inputs, data_samples = data['inputs'],data['right_inputs'], data['data_samples']

        # update metainfo since the image shape might change
        batch_input_shape = tuple(left_inputs[0].size()[-2:])
        for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
            data_sample.set_metainfo({
                'batch_input_shape': batch_input_shape,
                'pad_shape': pad_shape
            })

        # apply batch augmentations
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': torch.concat([left_inputs,right_inputs],dim=-1), 'data_samples': data_samples}
    
    
    
    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape
        
        
    def _forward(self, data: dict, training: bool = False, inputs_key='inputs') -> Union[dict, list]:
        """Performs normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        
        _batch_inputs = data[inputs_key]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data[inputs_key] = batch_inputs
        data.setdefault('data_samples', None)
        return data