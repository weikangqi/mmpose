# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MessageHub, MMLogger, print_log
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import logging
from mmpose.registry import METRICS
from mmpose.structures.bbox import bbox_xyxy2xywh
from mmpose.evaluation.functional import (oks_nms, soft_oks_nms, transform_ann, transform_pred,
                          transform_sigmas)

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmpose.evaluation.metrics import CocoMetric
from torch import Tensor
from mmengine.structures import BaseDataElement
from typing import Any, List, Optional, Sequence, Union
@METRICS.register_module()
class StereoCocoMetric(CocoMetric):

    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 pred_converter: Dict = None,
                 gt_converter: Dict = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(
                 ann_file,
                 use_area,
                 iou_type,
                 score_mode,
                 keypoint_score_thr,
                 nms_mode,
                 nms_thr,
                 format_only,
                 pred_converter,
                 gt_converter,
                 outfile_prefix,
                 collect_device,
                 prefix
            
        )
  

    # def process(self, data_batch: Sequence[dict],
    #             data_samples: Sequence[dict]) -> None:
    #     mid = len(data_samples)// 2 
    #     super().process(data_batch,data_samples[:mid])
    #     super().process(data_batch,data_samples[mid:])
    #     print("dd")
    # def compute_metrics(self, results: list) -> Dict[str, float]:
    #     super().compute_metrics(results)
        # super().compute_metrics(results[0])
    def evaluatexx(self, size: int) -> dict:
        if len(self.results_left) == 0 or len(self.results_right) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results_left = collect_results(
                self.results_left,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
            results_right = collect_results(
                self.results_right,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results_left = collect_results(self.results_left, size, self.collect_device)
            results_right = collect_results(self.results_right, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results_left = _to_cpu(results_left)
            results_right = _to_cpu(results_right)
            _metrics_left = self.compute_metrics(results_left)  # type: ignore
            _metrics_right = self.compute_metrics(results_right)
            
            # Add prefix to metric names
            if self.prefix:
                _metrics_left = {
                    'left_'+'/'.join((self.prefix, k)): v
                    for k, v in _metrics_left.items()
                }
                _metrics_right = {
                    'right_'+'/'.join((self.prefix, k)): v
                    for k, v in _metrics_right.items()
                }
            metrics_left = [_metrics_left]
            metrics_right = [_metrics_right]
        else:
            metrics_left = [None]  # type: ignore
            metrics_right = [None]
            
        metrics = [{**metrics_left[0] ,**metrics_right[0]}]
        broadcast_object_list(metrics)

        
        
        # reset the results list
        self.results_left.clear()
        self.results_right.clear()
        return metrics[0]
def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data