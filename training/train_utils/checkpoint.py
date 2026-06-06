# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import logging
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


def save_model_release_checkpoint(
    *,
    checkpoint_path: str,
    model: nn.Module,
    epoch: int,
    steps: Mapping[str, int],
    time_elapsed: float,
    model_ema: Optional[Any] = None,
    use_ema: bool = True,
) -> str:
    """Save a compact final checkpoint for inference/release."""
    model = unwrap_model(model)
    use_ema_weights = use_ema and model_ema is not None
    weight_source = "ema" if use_ema_weights else "raw"
    model_context = (
        model_ema.average_parameters(model)
        if use_ema_weights
        else contextlib.nullcontext()
    )

    with model_context:
        checkpoint = {
            "checkpoint_type": "final_release",
            "weight_source": weight_source,
            "prev_epoch": epoch,
            "steps": dict(steps),
            "time_elapsed": time_elapsed,
            "model": model.state_dict(),
        }
        if use_ema_weights:
            checkpoint["ema_steps"] = model_ema.num_updates
        robust_torch_save(checkpoint, checkpoint_path)

    return weight_source


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