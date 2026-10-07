# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math

import numpy as np
import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        axis_base_scales: tuple[float, float] = (1.0, 10.0),
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")
        if len(axis_base_scales) != 2:
            raise ValueError("`axis_base_scales` must contain exactly two values for height and width.")
        axis_base_scales = tuple(float(scale) for scale in axis_base_scales)
        if any(scale <= 0.0 for scale in axis_base_scales):
            raise ValueError("All `axis_base_scales` values must be positive.")
        
        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.axis_base_scales = axis_base_scales
        self.D_head = D_head

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(2, D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int, num_prefix_tokens: int = 0) -> tuple[Tensor, Tensor]:
        """Return sin/cos for prefix tokens at position zero followed by the patch grid."""
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        coords_h = torch.arange(H, **dd)  # [H]
        coords_w = torch.arange(W, **dd)  # [W]
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        if num_prefix_tokens:
            # Zero coordinates make RoPE an identity for class/register tokens.
            coords = torch.cat((coords.new_zeros(num_prefix_tokens, 2), coords), dim=0)

        # Prepare angles and sin/cos
        angles = coords[:, :, None] / self.periods[None, :, :]  # [N, 2, D//4], N = prefix + HW
        angles = angles.flatten(1, 2)  # [N, D//2]
        angles = angles.tile(2)  # [N, D]
        cos = torch.cos(angles)  # [N, D]
        sin = torch.sin(angles)  # [N, D]

        return (sin, cos)  # 2 * [N, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        num_periods = self.D_head // 4
        if self.base is not None:
            exponents = 2 * torch.arange(num_periods, device=device, dtype=dtype) / (self.D_head // 2)
            axis_bases = torch.tensor(self.axis_base_scales, device=device, dtype=dtype) * self.base
            periods = axis_bases[:, None] ** exponents[None, :]  # [2, D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, num_periods, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
            periods = periods.expand(2, -1).clone()  # [2, D//4]

        with torch.no_grad():
            self.periods.copy_(periods)
