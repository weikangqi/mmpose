from turtle import forward
from mmpose.models.heads import RTMOHead
from mmpose.registry import MODELS
from mmengine.model import BaseModule
from mmpose.utils.typing import ConfigType, OptConfigType
from typing import List, Optional, Tuple
from typing import Dict, List, Optional, Sequence, Tuple, Union
@MODELS.register_module()
class StereoRTMO_Head(BaseModule):
    def __init__(        self,
        num_keypoints: int,
        head_module_cfg: ConfigType,
        featmap_strides: Sequence[int] = [16, 32],
        num_classes: int = 1,
        use_aux_loss: bool = False,
        proxy_target_cc: bool = False,
        assigner: ConfigType = None,
        prior_generator: ConfigType = None,
        bbox_padding: float = 1.25,
        overlaps_power: float = 1.0,
        dcc_cfg: Optional[ConfigType] = None,
        loss_cls: Optional[ConfigType] = None,
        loss_bbox: Optional[ConfigType] = None,
        loss_oks: Optional[ConfigType] = None,
        loss_vis: Optional[ConfigType] = None,
        loss_mle: Optional[ConfigType] = None,
        loss_bbox_aux: Optional[ConfigType] = None,) -> None:
        super().__init__()
        self.left_head = RTMOHead( 
                            num_keypoints,
                            head_module_cfg,
                            featmap_strides,
                            num_classes,
                            use_aux_loss,
                            proxy_target_cc,
                            assigner,
                            prior_generator,
                            bbox_padding,
                            overlaps_power,
                            dcc_cfg,
                            loss_cls,
                            loss_bbox,
                            loss_oks,
                            loss_vis,
                            loss_mle,
                            loss_bbox_aux
        )
        self.right_head = RTMOHead(            
                            num_keypoints,
                            head_module_cfg,
                            featmap_strides,
                            num_classes,
                            use_aux_loss,
                            proxy_target_cc,
                            assigner,
                            prior_generator,
                            bbox_padding,
                            overlaps_power,
                            dcc_cfg,
                            loss_cls,
                            loss_bbox,
                            loss_oks,
                            loss_vis,
                            loss_mle,
                            loss_bbox_aux
            )
    def predict(self,feats,batch_data_samples,test_cfg):
        left_out = self.left_head.predict(feats[0],batch_data_samples[0],test_cfg)
        right_out = self.right_head.predict(feats[1],batch_data_samples[1],test_cfg)
        return [left_out,right_out]
        
    def loss(self,feats,batch_data_samples,train_cfg):
        # 需要处理batch_data_samples
        
        left_loss = self.left_head.loss(feats[0],batch_data_samples[0],train_cfg)
        right_loss = self.right_head.loss(feats[1],batch_data_samples[1],train_cfg)
        
        return self.merge(left_loss,right_loss)  # dict
        # return left_loss
    def merge(self,left_loss,right_loss):
        left_loss = {"left_" + key: value for key, value in left_loss.items()}
        # 为 dict2 的键添加前缀
        right_loss = {"right_" + key: value for key, value in right_loss.items()}
        # 合并两个字典
        merged_loss = {**left_loss, **right_loss}

        # merged_loss  = {k: (left_loss.get(k, 0) + right_loss.get(k, 0))/2 for k in left_loss.keys() and right_loss.keys()}

        return merged_loss

        
