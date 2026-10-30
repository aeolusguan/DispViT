# ------------------------------------------------------------------------
# DispViT (https://github.com/aeolusguan/DispViT)
# Copyright (c) 2026 CUHK T-Stone Robotics Institute. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Evaluate released models without constructing a Trainer or training datasets."""

from contextlib import ExitStack
from datetime import timedelta
from itertools import islice
import logging
import os
import random

import hydra
from hydra.utils import get_class, instantiate
import numpy as np
import torch
from torch import distributed as dist

from training.data.datasets import build_val_dataset
from training.data.worker_fn import get_rank
from training.eval_utils import build_evaluators, print_csv_format, validate_evaluation_config


@torch.inference_mode()
def evaluate(cfg):
    """Load a release checkpoint and report metrics for the selected benchmarks."""
    dataset_specs = validate_evaluation_config(cfg.eval, cfg.data.val_datasets)
    if not dataset_specs:
        raise ValueError("Select at least one evaluation dataset in data.val_datasets.")
    if not cfg.pretrained:
        raise ValueError("Set pretrained to a local checkpoint or Hugging Face model ID.")
    amp_types = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if cfg.amp.amp_dtype not in amp_types:
        raise ValueError(f"Invalid AMP dtype: {cfg.amp.amp_dtype}")

    device = torch.device(cfg.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        torch.backends.cudnn.deterministic = cfg.cuda.cudnn_deterministic
        torch.backends.cudnn.benchmark = cfg.cuda.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cfg.cuda.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.cuda.allow_tf32

    with ExitStack() as cleanup:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend=cfg.distributed.backend or ("nccl" if device.type == "cuda" else "gloo"),
                timeout=timedelta(minutes=cfg.distributed.timeout_mins),
            )
            cleanup.callback(dist.destroy_process_group)

        logging.basicConfig(
            level=logging.INFO if get_rank() == 0 else logging.WARNING,
            format="%(message)s",
        )
        random.seed(cfg.seed_value)
        np.random.seed(cfg.seed_value)
        torch.manual_seed(cfg.seed_value)

        model_class = get_class(cfg.model._target_)
        model = model_class.from_pretrained(cfg.pretrained).to(device).eval()
        evaluators = build_evaluators(cfg.eval, cfg.data.val_datasets)
        results = {}
        for (dataset_name, _), evaluator in zip(dataset_specs, evaluators):
            dataset = build_val_dataset(dataset_name)
            loader = instantiate(cfg.data.val, dataset, mode="val").get_loader(epoch=0)
            evaluator.reset()
            batches = loader if cfg.limit_val_batches is None else islice(loader, cfg.limit_val_batches)
            for index, batch in enumerate(batches):
                batch = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                with torch.autocast(
                    device.type,
                    enabled=cfg.amp.enabled and device.type == "cuda",
                    dtype=amp_types[cfg.amp.amp_dtype],
                ):
                    predictions = model(batch)
                evaluator.process(batch, predictions)
                if get_rank() == 0 and index % cfg.log_freq == 0:
                    logging.info("%s: %d/%d batches", dataset_name, index + 1, len(loader))

            metrics = evaluator.evaluate()
            results[dataset_name] = metrics
            if metrics:
                logging.info("Evaluation results for %s in csv format:", dataset_name)
                print_csv_format(metrics)
        return results


@hydra.main(version_base=None, config_path="config", config_name="zero_shot_eval")
def main(cfg):
    return evaluate(cfg)


if __name__ == "__main__":
    main()
