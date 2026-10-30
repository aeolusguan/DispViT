# ------------------------------------------------------------------------
# DispViT (https://github.com/aeolusguan/DispViT)
# Copyright (c) 2026 CUHK T-Stone Robotics Institute. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import importlib
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from .dispvit import InputPadder, shift_along_width
from .dpt_head import DPTHead


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DPT_MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
_DPT_INTERMEDIATE_LAYER_IDX = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
    "vitg": [9, 19, 29, 39],
}


def _build_module(module: Any, *, name: str) -> nn.Module:
    if module is None:
        raise ValueError(f"{name} must be provided.")
    if isinstance(module, nn.Module):
        return module
    return instantiate(module, _recursive_=False)


class _FrozenTokenProbeBase(nn.Module):
    """Shared probe utilities for frozen ViT patch tokens."""

    def __init__(
        self,
        *,
        num_classes: int = 19,
        n_last_layers: int = 4,
        embed_dim: int,
        patch_size: int,
        ignore_index: int = 255,
        eval_crop_size: tuple[int, int] | None = None,
        eval_stride: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.n_last_layers = int(n_last_layers)
        self.patch_size = int(patch_size)
        self.ignore_index = int(ignore_index)
        self.eval_crop_size = tuple(eval_crop_size) if eval_crop_size is not None else None
        self.eval_stride = tuple(eval_stride) if eval_stride is not None else self.eval_crop_size
        self.classifier = nn.Conv2d(
            embed_dim * self.n_last_layers,
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _linear_probe(self, features: tuple[torch.Tensor, ...], output_size: tuple[int, int]) -> torch.Tensor:
        if len(features) != self.n_last_layers:
            raise ValueError(f"Expected {self.n_last_layers} feature maps, got {len(features)}")
        feature = torch.cat([x.detach() for x in features], dim=1)
        logits = self.classifier(feature)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    @staticmethod
    def _window_starts(size: int, window: int, stride: int) -> list[int]:
        if size <= window:
            return [0]
        starts = list(range(0, size - window + 1, stride))
        last = size - window
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _should_slide(self, batch: Mapping[str, torch.Tensor]) -> bool:
        if self.training or self.eval_crop_size is None:
            return False
        height, width = batch["img1"].shape[-2:]
        crop_h, crop_w = self.eval_crop_size
        return height > crop_h or width > crop_w

    def _sliding_forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        height, width = batch["img1"].shape[-2:]
        crop_h = min(self.eval_crop_size[0], height)
        crop_w = min(self.eval_crop_size[1], width)
        stride_h = min(self.eval_stride[0], crop_h)
        stride_w = min(self.eval_stride[1], crop_w)
        y_starts = self._window_starts(height, crop_h, stride_h)
        x_starts = self._window_starts(width, crop_w, stride_w)

        logits_sum = None
        count = None
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = y0 + crop_h, x0 + crop_w
                crop_batch = dict(batch)
                crop_batch["img1"] = batch["img1"][..., y0:y1, x0:x1]
                if "img2" in batch:
                    crop_batch["img2"] = batch["img2"][..., y0:y1, x0:x1]
                crop_logits = self._forward_once(crop_batch)["sem_logits"]
                if logits_sum is None:
                    logits_sum = crop_logits.new_zeros(crop_logits.shape[0], crop_logits.shape[1], height, width)
                    count = crop_logits.new_zeros(crop_logits.shape[0], 1, height, width)
                logits_sum[..., y0:y1, x0:x1] += crop_logits
                count[..., y0:y1, x0:x1] += 1
        return {"sem_logits": logits_sum / count.clamp_min(1)}


class DispViTLastFourSemanticProbe(_FrozenTokenProbeBase):
    """DINOv2-style last-four-layer semantic probe for frozen DispViT."""

    def __init__(
        self,
        backbone: Any,
        *,
        encoder_type: str | None = None,
        init_weights: bool | None = None,
        num_classes: int = 19,
        n_last_layers: int = 4,
        ignore_index: int = 255,
        eval_crop_size: tuple[int, int] | None = None,
        eval_stride: tuple[int, int] | None = None,
        **kwargs_ignored,
    ) -> None:
        del encoder_type, init_weights, kwargs_ignored
        backbone = _build_module(backbone, name="backbone")
        super().__init__(
            num_classes=num_classes,
            n_last_layers=n_last_layers,
            embed_dim=backbone.pretrained.embed_dim,
            patch_size=backbone.pretrained.patch_size,
            ignore_index=ignore_index,
            eval_crop_size=eval_crop_size,
            eval_stride=eval_stride,
        )
        self.backbone = backbone
        self._freeze(self.backbone)

    def train(self, mode: bool = True) -> "DispViTLastFourSemanticProbe":
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_tokens(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shift_unit = 192 // self.backbone.groups
        shifts = [i * shift_unit for i in range(self.backbone.groups)]
        img = torch.cat([img1] + shift_along_width(img2, shifts), dim=1)
        return self.backbone.pretrained.get_intermediate_layers(
            img,
            n=self.n_last_layers,
            reshape=True,
            return_class_token=False,
            norm=True,
        )

    def _forward_once(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.backbone.eval()
        img1 = (batch["img1"] / 255.0 - self.backbone._resnet_mean) / self.backbone._resnet_std
        img2 = (batch["img2"] / 255.0 - self.backbone._resnet_mean) / self.backbone._resnet_std
        padder = InputPadder(img1.shape, mode="nmrf", divis_by=self.backbone.pretrained.patch_size)
        img1, img2 = padder.pad(img1, img2)

        with torch.no_grad():
            features = self._extract_tokens(img1, img2)

        logits = self._linear_probe(features, output_size=img1.shape[-2:])
        logits = padder.unpad(logits)
        return {"sem_logits": logits}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._should_slide(batch):
            return self._sliding_forward(batch)
        return self._forward_once(batch)


class DINOv2LastFourSemanticProbe(_FrozenTokenProbeBase):
    """Original DINOv2 baseline using the same last-four-layer probe protocol."""

    def __init__(
        self,
        encoder_type: str = "vitg",
        *,
        backbone: Any | None = None,
        init_weights: bool = True,
        num_classes: int = 19,
        n_last_layers: int = 4,
        ignore_index: int = 255,
        eval_crop_size: tuple[int, int] | None = None,
        eval_stride: tuple[int, int] | None = None,
        **kwargs_ignored,
    ) -> None:
        del kwargs_ignored
        if backbone is None:
            dinov2_model = getattr(
                importlib.import_module("..depth_anything.dinov2.hub.backbones", __package__),
                f"dinov2_{encoder_type}14",
            )
            backbone = dinov2_model(pretrained=init_weights)
        else:
            backbone = _build_module(backbone, name="backbone")

        super().__init__(
            num_classes=num_classes,
            n_last_layers=n_last_layers,
            embed_dim=backbone.embed_dim,
            patch_size=backbone.patch_size,
            ignore_index=ignore_index,
            eval_crop_size=eval_crop_size,
            eval_stride=eval_stride,
        )
        self.backbone = backbone
        self._freeze(self.backbone)

        for name, value in (("_mean", _IMAGENET_MEAN), ("_std", _IMAGENET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "DINOv2LastFourSemanticProbe":
        super().train(mode)
        self.backbone.eval()
        return self

    def _preprocess_left(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, InputPadder]:
        img1 = (batch["img1"] / 255.0 - self._mean) / self._std
        padder = InputPadder(img1.shape, mode="nmrf", divis_by=self.patch_size)
        (img1,) = padder.pad(img1)
        return img1, padder

    def _forward_once(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.backbone.eval()
        img1, padder = self._preprocess_left(batch)

        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                img1,
                n=self.n_last_layers,
                reshape=True,
                return_class_token=False,
                norm=True,
            )

        logits = self._linear_probe(features, output_size=img1.shape[-2:])
        logits = padder.unpad(logits)
        return {"sem_logits": logits}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._should_slide(batch):
            return self._sliding_forward(batch)
        return self._forward_once(batch)


class _FrozenDPTProbeBase(_FrozenTokenProbeBase):
    """Shared DPT decoder probe utilities for frozen ViT patch tokens."""

    def __init__(
        self,
        *,
        num_classes: int = 19,
        n_last_layers: int | None = None,
        embed_dim: int,
        patch_size: int,
        encoder_type: str,
        layer_indices: list[int] | tuple[int, ...] | None = None,
        ignore_index: int = 255,
        eval_crop_size: tuple[int, int] | None = None,
        eval_stride: tuple[int, int] | None = None,
        dpt_features: int | None = None,
        dpt_out_channels: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        encoder_type = str(encoder_type).lower()
        if layer_indices is None:
            if encoder_type not in _DPT_INTERMEDIATE_LAYER_IDX:
                raise ValueError(f"layer_indices are required for encoder_type={encoder_type}.")
            layer_indices = _DPT_INTERMEDIATE_LAYER_IDX[encoder_type]
        layer_indices = tuple(int(index) for index in layer_indices)
        if len(layer_indices) != 4:
            raise ValueError("The DPT semantic probe expects exactly four intermediate layers.")
        if n_last_layers is not None and int(n_last_layers) != len(layer_indices):
            raise ValueError("n_last_layers must match the number of DPT layer_indices when provided.")

        super().__init__(
            num_classes=num_classes,
            n_last_layers=len(layer_indices),
            embed_dim=embed_dim,
            patch_size=patch_size,
            ignore_index=ignore_index,
            eval_crop_size=eval_crop_size,
            eval_stride=eval_stride,
        )
        del self.classifier
        self.layer_indices = list(layer_indices)

        if dpt_features is None or dpt_out_channels is None:
            if encoder_type not in _DPT_MODEL_CONFIGS:
                raise ValueError(f"dpt_features and dpt_out_channels are required for encoder_type={encoder_type}.")
            defaults = _DPT_MODEL_CONFIGS[encoder_type]
            if dpt_features is None:
                dpt_features = defaults["features"]
            if dpt_out_channels is None:
                dpt_out_channels = defaults["out_channels"]

        self.dpt_head = DPTHead(
            dim_in=embed_dim,
            patch_size=self.patch_size,
            output_dim=self.num_classes,
            features=int(dpt_features),
            out_channels=list(dpt_out_channels),
        )

    def _dpt_probe(
        self,
        features: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        patch_h: int,
        patch_w: int,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != self.n_last_layers:
            raise ValueError(f"Expected {self.n_last_layers} feature maps, got {len(features)}")
        logits = self.dpt_head(list(features), patch_h=patch_h, patch_w=patch_w)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DispViTDPTSemanticProbe(_FrozenDPTProbeBase):
    """Frozen DispViT semantic probe with a trainable DPT dense decoder."""

    def __init__(
        self,
        backbone: Any,
        *,
        encoder_type: str | None = None,
        num_classes: int = 19,
        n_last_layers: int | None = None,
        layer_indices: list[int] | tuple[int, ...] | None = None,
        init_weights: bool | None = None,
        ignore_index: int = 255,
        eval_crop_size: tuple[int, int] | None = None,
        eval_stride: tuple[int, int] | None = None,
        dpt_features: int | None = None,
        dpt_out_channels: list[int] | tuple[int, ...] | None = None,
        **kwargs_ignored,
    ) -> None:
        del init_weights, kwargs_ignored
        backbone = _build_module(backbone, name="backbone")
        encoder_type = encoder_type or getattr(backbone, "encoder_type", "")
        super().__init__(
            num_classes=num_classes,
            n_last_layers=n_last_layers,
            embed_dim=backbone.pretrained.embed_dim,
            patch_size=backbone.pretrained.patch_size,
            encoder_type=encoder_type,
            layer_indices=layer_indices,
            ignore_index=ignore_index,
            eval_crop_size=eval_crop_size,
            eval_stride=eval_stride,
            dpt_features=dpt_features,
            dpt_out_channels=dpt_out_channels,
        )
        self.backbone = backbone
        self._freeze(self.backbone)

    def train(self, mode: bool = True) -> "DispViTDPTSemanticProbe":
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_dpt_tokens(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        shift_unit = 192 // self.backbone.groups
        shifts = [i * shift_unit for i in range(self.backbone.groups)]
        img = torch.cat([img1] + shift_along_width(img2, shifts), dim=1)
        return self.backbone.pretrained.get_intermediate_layers(
            img,
            n=self.layer_indices,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

    def _forward_once(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.backbone.eval()
        img1 = (batch["img1"] / 255.0 - self.backbone._resnet_mean) / self.backbone._resnet_std
        img2 = (batch["img2"] / 255.0 - self.backbone._resnet_mean) / self.backbone._resnet_std
        padder = InputPadder(img1.shape, mode="nmrf", divis_by=self.backbone.pretrained.patch_size)
        img1, img2 = padder.pad(img1, img2)
        patch_h, patch_w = img1.shape[-2] // self.patch_size, img1.shape[-1] // self.patch_size

        with torch.no_grad():
            features = self._extract_dpt_tokens(img1, img2)

        logits = self._dpt_probe(features, patch_h=patch_h, patch_w=patch_w, output_size=img1.shape[-2:])
        logits = padder.unpad(logits)
        return {"sem_logits": logits}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._should_slide(batch):
            return self._sliding_forward(batch)
        return self._forward_once(batch)


class DINOv2DPTSemanticProbe(_FrozenDPTProbeBase):
    """Original DINOv2 semantic probe with a trainable DPT dense decoder."""

    def __init__(
        self,
        encoder_type: str = "vitg",
        *,
        backbone: Any | None = None,
        init_weights: bool = True,
        num_classes: int = 19,
        n_last_layers: int | None = None,
        layer_indices: list[int] | tuple[int, ...] | None = None,
        ignore_index: int = 255,
        eval_crop_size: tuple[int, int] | None = None,
        eval_stride: tuple[int, int] | None = None,
        dpt_features: int | None = None,
        dpt_out_channels: list[int] | tuple[int, ...] | None = None,
        **kwargs_ignored,
    ) -> None:
        del kwargs_ignored
        if backbone is None:
            dinov2_model = getattr(
                importlib.import_module("..depth_anything.dinov2.hub.backbones", __package__),
                f"dinov2_{encoder_type}14",
            )
            backbone = dinov2_model(pretrained=init_weights)
        else:
            backbone = _build_module(backbone, name="backbone")

        super().__init__(
            num_classes=num_classes,
            n_last_layers=n_last_layers,
            embed_dim=backbone.embed_dim,
            patch_size=backbone.patch_size,
            encoder_type=encoder_type,
            layer_indices=layer_indices,
            ignore_index=ignore_index,
            eval_crop_size=eval_crop_size,
            eval_stride=eval_stride,
            dpt_features=dpt_features,
            dpt_out_channels=dpt_out_channels,
        )
        self.backbone = backbone
        self._freeze(self.backbone)

        for name, value in (("_mean", _IMAGENET_MEAN), ("_std", _IMAGENET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "DINOv2DPTSemanticProbe":
        super().train(mode)
        self.backbone.eval()
        return self

    def _preprocess_left(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, InputPadder]:
        img1 = (batch["img1"] / 255.0 - self._mean) / self._std
        padder = InputPadder(img1.shape, mode="nmrf", divis_by=self.patch_size)
        (img1,) = padder.pad(img1)
        return img1, padder

    def _forward_once(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.backbone.eval()
        img1, padder = self._preprocess_left(batch)
        patch_h, patch_w = img1.shape[-2] // self.patch_size, img1.shape[-1] // self.patch_size

        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                img1,
                n=self.layer_indices,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        logits = self._dpt_probe(features, patch_h=patch_h, patch_w=patch_w, output_size=img1.shape[-2:])
        logits = padder.unpad(logits)
        return {"sem_logits": logits}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._should_slide(batch):
            return self._sliding_forward(batch)
        return self._forward_once(batch)

