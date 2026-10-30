# ------------------------------------------------------------------------
# DispViT (https://github.com/aeolusguan/DispViT)
# Copyright (c) 2026 CUHK T-Stone Robotics Institute. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import einops

from dataclasses import dataclass


@dataclass(eq=False)
class DispLoss(torch.nn.Module):
    def __init__(self, disp, logits):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.disp = disp
        self.logits = logits

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        gt_disp = batch["disp"]
        valid = torch.logical_and(batch["valid"], gt_disp < self.disp["max_disp"])

        loss_disp = disp_loss(predictions["disp"], gt_disp, valid)
        loss_logits = disp_softmax(predictions["disp_logits"], gt_disp, valid)

        loss_dict = {
            "objective": self.disp["weight"]*loss_disp + self.logits["weight"]*loss_logits,
            "loss_disp": loss_disp,
            "loss_logits": loss_logits,
        }
        return loss_dict


def disp_loss(pred_disp, target_disp, mask):
    pred_disp = pred_disp.float()
    target_disp = target_disp.float()
    error = torch.abs(pred_disp[mask] - target_disp[mask])
    loss = error.sum() / (mask.sum() + 1e-6)
    return loss


def disp_softmax(logits, labels, mask):
    logits = logits.float()
    labels = labels.float()
    labels = labels.clamp(0, 381.0)
    interval = 381.0 / 127.0
    labels = labels.unsqueeze(1) / interval
    lb = torch.floor(labels).long()
    hb = (1 + lb).clamp_max(127)
    labels_ = torch.zeros_like(logits)
    wh = labels - lb
    labels_.scatter_add_(1, lb, 1.0 - wh)
    labels_.scatter_add_(1, hb, wh)
    loss = F.cross_entropy(logits, labels_, reduction='none')
    mask = mask.float()
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss


@dataclass(eq=False)
class RefineLoss(torch.nn.Module):
    def __init__(self, disp, loss_type, **kwargs_ignored):
        super().__init__()
        self.disp = disp
        self.weights = disp["weight"]
        if loss_type not in ['l1', 'smooth_l1']:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.criterion = F.l1_loss
        else:
            self.criterion = F.smooth_l1_loss

    def _loss_fn(self, pred, target, mask, conf=None, alpha=0.2):
        pred, target = pred[mask], target[mask]
        conf = conf[mask] if conf is not None else None

        if pred.numel() == 0:
            return pred.sum() if conf is None else pred.sum() + conf.sum()
        if conf is None:
            return self.criterion(pred, target, reduction='mean')

        error = torch.abs(pred - target)
        return (error * conf - alpha * torch.log(conf)).mean()

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        gt_disp = batch["disp"]
        valid = torch.logical_and(batch["valid"], gt_disp < self.disp["max_disp"])

        disp_preds = predictions["disp_all"]
        confs = predictions.get("conf_all")
        if confs is None:
            loss_disp = [self._loss_fn(pred, gt_disp, valid) for pred in disp_preds]
        else:
            if confs.shape != disp_preds.shape:
                raise ValueError("conf_all and disp_all must have the same shape.")
            loss_disp = [
                self._loss_fn(pred, gt_disp, valid, conf=conf)
                for pred, conf in zip(disp_preds, confs)
            ]
        assert len(loss_disp) == len(self.weights), f"Expected {len(self.weights)} disparity predictions, but got {len(loss_disp)}"

        objective = sum(w * l for w, l in zip(self.weights, loss_disp))
        loss_dict = {
            "objective": objective,
            "loss_disp": loss_disp[-1],
            **{f"loss_disp_{i}": l for i, l in enumerate(loss_disp)},
        }

        return loss_dict
