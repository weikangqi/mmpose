from itertools import zip_longest
from typing import List, Optional, Union

from mmengine.utils import is_list_of
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from mmpose.models.pose_estimators.bottomup import BottomupPoseEstimator



@MODELS.register_module()
class StereoBottomupPoseEstimator(BottomupPoseEstimator):
    
    
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 use_syncbn: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            use_syncbn=use_syncbn,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.baseline = 0.02
        self.focal = 320
        
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W*2).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        # TODO 
        n,c,h,w = inputs.shape
        
        left_input = inputs[:,:,:,:w // 2]
        right_input = inputs[:,:,:,w // 2:]
            
        # else:
        # 正常的COCO的数据集
        left_feats = self.extract_feat([left_input,right_input])
        # right_feats = self.extract_feat(right_input)

        losses = dict()

        if self.with_head:
            losses.update(
                self.head.loss(left_feats, data_samples, train_cfg=self.train_cfg))

        return losses
    
    def predict(self, inputs: Union[Tensor, List[Tensor]],
                data_samples: SampleList) -> SampleList:

        assert self.with_head, (
            'The model must have head to perform prediction.')

        # multiscale_test = self.test_cfg.get('multiscale_test', False)
        # flip_test = self.test_cfg.get('flip_test', False)

        # enable multi-scale test
        # data_samples = data_samples[0]
        # aug_scales = data_samples[0].metainfo.get('aug_scales', None)
        # if multiscale_test:
        #     assert isinstance(aug_scales, list)
        #     assert is_list_of(inputs, Tensor)
        #     # `inputs` includes images in original and augmented scales
        #     # assert len(inputs) == len(aug_scales) + 1
        # else:
        #     assert isinstance(inputs, Tensor)
        #     # single-scale test
        #     inputs = [inputs]

        feats = []
        
        n,c,h,w = inputs.shape
        
        left_input = inputs[:,:,:,:w // 2]
        right_input = inputs[:,:,:,w // 2:]
        
        # feats_right = []
        # for _inputs in inputs:
        #     if flip_test:
        #         _feats_orig = self.extract_feat(_inputs)
        #         _feats_flip = self.extract_feat(_inputs.flip(-1))
        #         _feats = [_feats_orig, _feats_flip]
        #     else:
        #         _feats = self.extract_feat(_inputs)

        #     feats.append(_feats)

        # if not multiscale_test:
        #     feats = feats[0]
        feats = self.extract_feat([left_input,right_input])
        preds_left,preds_right = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds_left, tuple):
            batch_pred_instances_left, batch_pred_fields_left = preds_left
        else:
            batch_pred_instances_left = preds_left
            batch_pred_fields_left = None
        
        if isinstance(preds_right, tuple):
            batch_pred_instances_right, batch_pred_fields_right = preds_right
        else:
            batch_pred_instances_right = preds_right
            batch_pred_fields_right = None

        results_left = self.add_pred_to_datasample(batch_pred_instances_left,
                                              batch_pred_fields_left, data_samples[0])
        results_right = self.add_pred_to_datasample(batch_pred_instances_right,
                                              batch_pred_fields_right, data_samples[1])

        return results_left+results_right
        # 合并
        # return [results_left , results_right]
    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
            The length of the list is the batch size when ``merge==False``, or
            1 when ``merge==True``.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            input_size = data_sample.metainfo['input_size']
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']

            # convert keypoint coordinates from input space to image space
            pred_instances.keypoints = pred_instances.keypoints / input_size \
                * input_scale + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            # convert bbox coordinates from input space to image space
            if 'bboxes' in pred_instances:
                bboxes = pred_instances.bboxes.reshape(
                    pred_instances.bboxes.shape[0], 2, 2)
                bboxes = bboxes / input_size * input_scale + input_center \
                    - 0.5 * input_scale
                pred_instances.bboxes = bboxes.reshape(bboxes.shape[0], 4)

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                data_sample.pred_fields = pred_fields

        return batch_data_samples
