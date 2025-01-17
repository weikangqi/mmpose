from mmpose.models.necks.hybrid_encoder import HybridEncoder
from mmpose.registry import MODELS
from mmengine.model import BaseModule
from mmpose.utils.typing import ConfigType, OptConfigType
from typing import List, Optional, Tuple
import torch
@MODELS.register_module()
class StereoHybridEncoder(BaseModule):
    def __init__(self,
                 encoder_cfg: ConfigType = dict(),
                 projector: OptConfigType = None,
                 num_encoder_layers: int = 1,
                 in_channels: List[int] = [512, 1024, 2048],
                 feat_strides: List[int] = [8, 16, 32],
                 hidden_dim: int = 256,
                 use_encoder_idx: List[int] = [2],
                 pe_temperature: int = 10000,
                 widen_factor: float = 1.0,
                 deepen_factor: float = 1.0,
                 spe_learnable: bool = False,
                 output_indices: Optional[List[int]] = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True)):
        super(StereoHybridEncoder, self).__init__()
        self.model = HybridEncoder(    encoder_cfg,
                                                    projector,
                                                    num_encoder_layers,
                                                    in_channels,
                                                    feat_strides,
                                                    hidden_dim,
                                                    use_encoder_idx,
                                                    pe_temperature,
                                                    widen_factor,
                                                    deepen_factor,
                                                    spe_learnable,
                                                    output_indices,
                                                    norm_cfg,
                                                    act_cfg)

    def forward(self, x,y):

        x = self.model.forward(x)
        with torch.no_grad():
            y = self.model.forward(y)
        return x
        # return left_out
    
            
            
        
        
    