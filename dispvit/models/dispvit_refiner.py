# Copyright (c) CUHK T-Stone Robotics Institute.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from .dpt_head import DPTHead


class RefinementBlendConv(nn.Module):
    """Patch projection for left RGB plus right-view refinement context."""

    def __init__(self, proj: nn.Conv2d, aux_channels: int = 4) -> None:
        super().__init__()
        self.proj = proj
        self.proj_aux = nn.Conv2d(
            in_channels=aux_channels,
            out_channels=proj.out_channels // 2,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            dilation=proj.dilation,
            groups=1,
            bias=False,
        )
        nn.init.zeros_(self.proj_aux.weight)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 7:
            raise ValueError(f"Expected 7 input channels, got {img.shape[1]}")
        left = img[:, :3].contiguous()
        aux = img[:, 3:].contiguous()
        left_features = self.proj(left)
        aux_features = self.proj_aux(aux)
        return left_features + torch.cat((torch.zeros_like(aux_features), aux_features), dim=1)


class DispViTRefiner(nn.Module):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    intermediate_layer_idx = {
        "vits": [2, 5, 8, 11],
        "vitb": [2, 5, 8, 11],
        "vitl": [4, 11, 17, 23],
        "vitg": [9, 19, 29, 39],
    }

    def __init__(
        self,
        regressor: Any,
        encoder_type: str = "vitb",
        disp_norm_scale: float = 384.0,
        freeze_regressor: bool = True,
        init_weights: bool = True,
        rope_axis_base_scales: tuple[float, float] | None = (1.0, 1.0),
    ) -> None:
        super().__init__()
        if encoder_type not in self.model_configs:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.encoder_type = encoder_type
        self.disp_norm_scale = float(disp_norm_scale)
        self.freeze_regressor = bool(freeze_regressor)

        self.regressor = self._build_regressor(regressor)
        if self.freeze_regressor:
            self._freeze_regressor()

        dinov2_model = getattr(
            importlib.import_module("..layers.dinov2", __package__),
            f"dinov2_{encoder_type}14",
        )
        backbone_kwargs = {}
        if rope_axis_base_scales is not None:
            backbone_kwargs["rope_axis_base_scales"] = rope_axis_base_scales
        self.pretrained = dinov2_model(pretrained=init_weights, **backbone_kwargs)
        self._build_patch_embed(self.pretrained.patch_embed)

        encoder = self.pretrained
        self.depth_head = DPTHead(
            encoder.embed_dim,
            features=self.model_configs[encoder_type]["features"],
            out_channels=self.model_configs[encoder_type]["out_channels"],
            patch_size=encoder.patch_size,
        )

    @staticmethod
    def _build_regressor(
        regressor: Any,
    ) -> nn.Module:
        if regressor is None:
            raise ValueError("DispViTRefiner requires a regressor.")
        if isinstance(regressor, nn.Module):
            return regressor
        return instantiate(regressor, _recursive_=False)

    def _freeze_regressor(self) -> None:
        self.regressor.eval()
        for param in self.regressor.parameters():
            param.requires_grad_(False)
        for buffer in self.regressor.buffers():
            buffer.requires_grad = False

    @staticmethod
    def _build_patch_embed(patch_embed: nn.Module) -> None:
        patch_embed.proj = RefinementBlendConv(patch_embed.proj, aux_channels=4)
        patch_embed.in_chans = 7

    def _zero_init_residual_head(self) -> None:
        output_layers = getattr(self.depth_head.scratch, "output_conv2", None)
        if output_layers is None:
            return
        for module in reversed(list(output_layers.modules())):
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return

    @staticmethod
    def warp_right_to_left(right: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        if disp.ndim == 3:
            disp = disp.unsqueeze(1)
        if disp.ndim != 4 or disp.shape[1] != 1:
            raise ValueError(f"Expected disparity shape [B,1,H,W] or [B,H,W], got {tuple(disp.shape)}")

        batch_size, _, height, width = right.shape
        device = right.device
        disp = disp[:, 0].to(device=device, dtype=torch.float32)

        xs = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width).expand(batch_size, height, width)
        ys = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1).expand(batch_size, height, width)

        sample_x = xs - disp
        sample_y = ys
        if width > 1:
            sample_x = 2.0 * sample_x / (width - 1) - 1.0
        else:
            sample_x = torch.zeros_like(sample_x)
        if height > 1:
            sample_y = 2.0 * sample_y / (height - 1) - 1.0
        else:
            sample_y = torch.zeros_like(sample_y)

        grid = torch.stack((sample_x, sample_y), dim=-1)
        return F.grid_sample(right, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        img1, img2, padder = self.regressor.preprocess_images(batch)
        if self.freeze_regressor:
            self.regressor.eval()
            with torch.no_grad():
                regression_output = self.regressor.forward_padded(img1, img2)
        else:
            regression_output = self.regressor.forward_padded(img1, img2)

        init_disp = regression_output["disp"].unsqueeze(1).float()

        refine_init_disp = init_disp.detach()
        warped_img2 = self.warp_right_to_left(img2, refine_init_disp)
        disp_context = (refine_init_disp / self.disp_norm_scale).to(dtype=img1.dtype)
        refine_input = torch.cat((img1, warped_img2, disp_context), dim=1)

        _, _, height, width = refine_input.shape
        patch_h = height // self.pretrained.patch_size
        patch_w = width // self.pretrained.patch_size
        features = self.pretrained.get_intermediate_layers(
            refine_input,
            self.intermediate_layer_idx[self.encoder_type],
            return_class_token=True,
        )

        with torch.autocast(device_type=refine_input.device.type, enabled=not self.training):
            disp = self.depth_head(features, patch_h, patch_w)

        disp = F.relu(disp)
        disp = padder.unpad(disp).squeeze(1)
        disp_init = padder.unpad(init_disp).squeeze(1)
        disp_logits = padder.unpad(regression_output["disp_logits"])

        out = {
            "disp": disp,
            "disp_all": disp.unsqueeze(0),
            "disp_regress": disp_init,
            "disp_logits": disp_logits,
        }
        return out
