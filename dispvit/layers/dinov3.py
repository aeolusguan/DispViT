# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import Optional, Union
from urllib.parse import urlparse
from pathlib import Path

import torch


DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"


class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")


def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov3_vit_model_arch(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
):
    if "plus" in compact_arch_name:
        model_arch = compact_arch_name.replace("plus", f"{patch_size}plus")
    else:
        model_arch = f"{compact_arch_name}{patch_size}"
    return model_arch


def _make_dinov3_vit_model_url(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    model_arch = _make_dinov3_vit_model_arch(patch_size=patch_size, compact_arch_name=compact_arch_name)
    version_suffix = f"_{version}" if version else ""
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""
    model_dir = f"{model_name}_{model_arch}"
    model_filename = f"{model_name}_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_vit(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    compact_arch_name: str = "vitb",
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    ffn_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_path_rate: float = 0.0,
    layerscale_init: float | None = None,
    ffn_layer: str = "mlp",
    ffn_bias: bool = True,
    proj_bias: bool = True,
    block_chunks: int = 0,
    n_storage_tokens: int = 0,
    mask_k_bias: bool = False,
    pretrained: bool = True,
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    check_hash: bool = False,
    **kwargs,
):
    from .vision_transformer import DinoVisionTransformer

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        init_values=layerscale_init,
        ffn_layer=ffn_layer,
        ffn_bias=ffn_bias,
        proj_bias=proj_bias,
        block_chunks=block_chunks,
        num_register_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
    )
    vit_kwargs.update(**kwargs)
    model = DinoVisionTransformer(**vit_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_vit_model_url(
                patch_size=patch_size,
                compact_arch_name=compact_arch_name,
                version=version,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.init_weights()
    return model


def dinov3_vitb16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitb",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,    
):
    untie_global_and_local_cls_norm = False
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "8aa4cbdd"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "eadcf0ff"
        untie_global_and_local_cls_norm = True
    elif type(weights) is str:
        import re

        pattern = r"-(.{8}).pth"
        matches = re.findall(pattern, weights)
        if len(matches) != 1:
            raise ValueError(f"Unexpected weights specification for the ViT-L backbone: {weights}")
        hash = matches[0]
        if hash == "eadcf0ff":
            untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitl",
        check_hash=check_hash,
        **kwargs,
    )