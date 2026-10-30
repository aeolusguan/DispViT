# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

import torch
import torch.nn as nn
import os
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from wcmatch import fnmatch





# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)




class DDPCheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.worker_id = rank
        self.epoch = epoch

    def save_checkpoint(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()

        if self.worker_id == 0:
            for ckpt_name in self.checkpoint_names:
                checkpoint_path = os.path.join(
                    self.checkpoint_folder, f"{ckpt_name}.pt"
                )
                logging.info(
                    f"Saving checkpoint at epoch {self.epoch} to {checkpoint_path}"
                )
                robust_torch_save(checkpoint, checkpoint_path)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def export_model_config(model_config: Mapping[str, Any], model: nn.Module) -> Dict[str, Any]:
    """Export constructor settings from the training config and loaded model."""
    model = unwrap_model(model)
    config = (
        OmegaConf.to_container(model_config, resolve=True)
        if OmegaConf.is_config(model_config)
        else deepcopy(dict(model_config))
    )
    target = config.pop("_target_", None)
    for key in ("_recursive_", "_convert_", "_partial_"):
        config.pop(key, None)

    dispvit_target = "dispvit.models.dispvit.DispViT"
    if target in (dispvit_target, f"{dispvit_target}.from_pretrained"):
        return {
            "encoder_type": model.encoder_type,
            "groups": model.groups,
            "init_weights": False,
        }

    if target == "dispvit.models.dispvit_plus.DispViTPlus":
        # Drop DispViT defaults inherited by the full-model training configs.
        for key in ("encoder_type", "groups", "init_weights"):
            config.pop(key, None)
        if config["regressor"].get("_target_") in (dispvit_target, f"{dispvit_target}.from_pretrained"):
            config["regressor"] = {
                "_target_": dispvit_target,
                **export_model_config(config["regressor"], model.regressor),
            }
        if config["extractor"].get("_target_") == "dispvit.models.extractor.ConvNextExtractor":
            config["extractor"]["_init_weights"] = False
    return config


def save_model_release_checkpoint(
    *,
    checkpoint_path: str,
    model: nn.Module,
    epoch: int,
    steps: Mapping[str, int],
    time_elapsed: float,
    model_config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Save the current model state in a compact checkpoint for inference/release."""
    model = unwrap_model(model)

    checkpoint = {
        "checkpoint_type": "final_release",
        "weight_source": "raw",
        "prev_epoch": epoch,
        "steps": dict(steps),
        "time_elapsed": time_elapsed,
        "model": model.state_dict(),
    }
    if model_config is not None:
        checkpoint["model_config"] = export_model_config(model_config, model)
    robust_torch_save(checkpoint, checkpoint_path)

    return "raw"


def prune_training_checkpoints(
    checkpoint_folder: str,
    keep_paths: Optional[List[str]] = None,
) -> List[str]:
    """Remove heavy resumable training checkpoints after final release export."""
    if not os.path.isdir(checkpoint_folder):
        return []

    keep_paths = {os.path.abspath(path) for path in keep_paths or []}
    removed_paths = []
    for filename in os.listdir(checkpoint_folder):
        is_training_checkpoint = filename == "checkpoint.pt" or (
            filename.startswith("checkpoint_") and filename.endswith(".pt")
        )
        if not is_training_checkpoint:
            continue

        checkpoint_path = os.path.join(checkpoint_folder, filename)
        if os.path.abspath(checkpoint_path) in keep_paths:
            continue
        if os.path.isfile(checkpoint_path):
            g_pathmgr.rm(checkpoint_path)
            removed_paths.append(checkpoint_path)

    return removed_paths


def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """
    A more robust version of torch.save that works better with preemptions
    and corruptions if a job is preempted during save.
    """
    # Move the existing checkpoint to a backup location
    backup_checkpoint_path = checkpoint_path + ".bak"
    backup_checkpoint_path_saved = False
    if g_pathmgr.exists(checkpoint_path):
        assert not g_pathmgr.exists(
            backup_checkpoint_path
        ), f"this should not exist... {backup_checkpoint_path}"
        g_pathmgr.mv(checkpoint_path, backup_checkpoint_path)
        backup_checkpoint_path_saved = True
    # Save the checkpoint
    with g_pathmgr.open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    # Remove the backup checkpoint
    if backup_checkpoint_path_saved:
        g_pathmgr.rm(backup_checkpoint_path)