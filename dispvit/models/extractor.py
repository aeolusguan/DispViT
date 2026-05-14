from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ConvNeXt
from torchvision.models.convnext import CNBlockConfig

from .dpt_head import _make_scratch, _make_fusion_block


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)


        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not (stride == 1 and in_planes == planes):
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        identity = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))

        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)


class ResNet(nn.Module):
    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 1/2
        self.norm1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1, norm_layer=norm_layer) # 1/2
        self.layer2 = self._make_layer(96, stride=2, norm_layer=norm_layer) # 1/4
        self.layer3 = self._make_layer(128, stride=1, norm_layer=norm_layer) # 1/4

        self.conv2 = nn.Conv2d(128, output_dim, 1, 1, 0)

        self.output_dim = output_dim

        self.apply(init_weights)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/4
        x = self.conv2(x)

        return x


class ConvNextExtractor(nn.Module):
    def __init__(
        self,
        output_dim,
        stochastic_depth_prob: float = 0.0,
        _init_weights: bool = True,
    ):
        super().__init__()
        
        model = ConvNeXt(
            block_setting=[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, None, 9),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.features = model.features

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.norms = nn.ModuleList([
            norm_layer(96),
            norm_layer(192),
            norm_layer(384),
        ])

        self.scratch = _make_scratch(
            [96, 192, 384],
            output_dim,
            expand=False,
        )

        # Attach additional modules to scratch
        self.scratch.stem_transpose=None
        self.scratch.refinenet1 = _make_fusion_block(output_dim)
        self.scratch.refinenet2 = _make_fusion_block(output_dim)
        self.scratch.refinenet3 = _make_fusion_block(output_dim, has_residual=False)

        self.output_dim = output_dim

        if _init_weights:
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            pretrained_dict = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        else:
            self.apply(init_weights)

    def _forward_features(self, x):
        outs = []
        for i, blk in enumerate(self.features):
            if self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i in [1, 3, 5]:  # after stage 1,2,3
                x = self.norms[len(outs)](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
                outs.append(x)
        return outs
    
    def forward(self, x):
        features = self._forward_features(x)

        # Fuse features from multiple layers.
        out = self.scratch_forward(features)

        return out

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)

        out = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])
        del layer_1_rn, layer_1

        return out    


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm)):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)