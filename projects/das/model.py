import os
import sys
sys.path.append('/workspace/mmpose3d')
sys.path.append('/workspace/mmpose3d/projects/das')
from mmpose.models.backbones import MSPN
from mmpose.models.necks import FPN
from pose_heads import DASHead
import torch
import torch.nn as nn




class DASModel(nn.Module):

    def __init__(self):
        super(DASModel,self).__init__()
        fpn_channels = 256
        num_joints = 15
        self.backbone = MSPN(
            unit_channels=256,
            num_stages=2,
            num_units=4,
            num_blocks=[3, 4, 6, 3],
            norm_cfg=dict(type='SyncBN'),
        )
        
        self.neck = FPN(
            in_channels=[256, 256, 256, 256],
            out_channels=fpn_channels,
            norm_cfg=dict(type='SyncBN'),
            add_extra_convs='on_output',
            start_level=1,
            num_outs=4,
        )
        
        self.head = DASHead(
            num_classes=1,
            in_channels=fpn_channels,
            feat_channels=fpn_channels,
            regress_ranges=((-1, 80), (80, 160), (160, 320), (320, 1e8),),
            strides=[8, 16, 32, 64],
            num_joints=num_joints,
            depth_factor=20,
            z_norm=50,
            root_idx=2,
            recursive_update=dict(
                num_joints=num_joints,
                prev_loss=True,
            num_heads=4,
            in_channels=256,
            feat_channels=256,
            num_layers=1,
            dim=3,
            ),
            center_sample_radius=1.5,
            cls_branch=(256, ),
            reg_branch=(
                (256, ),
                (256, ),
                (256, ),
                (256, ),
            ),
            centerness_on_reg=True,
            conv_bias=True,
            dcn_on_last_conv=True,
        )


    
    def forward(self,x):
        feat = self.backbone(x)[-1]
        feat = self.neck(feat)
        out = self.head(feat)
        return out
        

        

if __name__=="__main__":
    
    x = torch.randn([1,3,640,640])
    model = DASModel().eval().cpu()
    out= model(x)

    torch.onnx.export(
        model,
        x,
        'das.onnx'
    )