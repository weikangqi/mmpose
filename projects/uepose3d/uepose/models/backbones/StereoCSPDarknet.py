from turtle import forward
from mmpose.models.backbones.csp_darknet import CSPDarknet
from mmengine.model import BaseModule
from mmpose.registry import MODELS
import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmpose.models.utils import CSPLayer

@MODELS.register_module()
class StereoCSPDarknet(BaseModule):
    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__()
        self.model = CSPDarknet(
               arch,
                 deepen_factor,
                 widen_factor,
                 out_indices,
                 frozen_stages,
                 use_depthwise,
                 arch_ovewrite,
                 spp_kernal_sizes,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 norm_eval,
                 init_cfg
                )
       
    
    def _freeze_stages(self):
        self.model._freeze_stages()

    
            
    def train(self, mode=True):
        self.model.train(mode)

    
    def forward(self,x,y):
        x = self.model.forward(x)
        with torch.no_grad():
            y = self.model.forward(y)
        return x,y
        # return left_out