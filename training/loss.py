import torch
import torch.nn.functional as F

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
    error = torch.abs(pred_disp - target_disp)
    loss = (error * mask).sum() / (mask.sum() + 1e-6)
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