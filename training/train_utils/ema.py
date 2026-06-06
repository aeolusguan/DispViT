import contextlib
from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


class ModelEMA:
    """Exponential moving average for trainable floating-point parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_steps: int = 0,
        update_every: int = 1,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        if update_after_steps < 0:
            raise ValueError(f"EMA update_after_steps must be >= 0, got {update_after_steps}")
        if update_every <= 0:
            raise ValueError(f"EMA update_every must be > 0, got {update_every}")

        self.decay = float(decay)
        self.update_after_steps = int(update_after_steps)
        self.update_every = int(update_every)
        self.num_updates = 0
        self.shadow = OrderedDict(
            (name, param.detach().clone())
            for name, param in self._averaged_parameters(model)
        )

    @staticmethod
    def _averaged_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, param in unwrap_model(model).named_parameters():
            if param.requires_grad and param.is_floating_point():
                yield name, param

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        for name, param in self._averaged_parameters(model):
            if name in self.shadow:
                self.shadow[name].copy_(param.detach())
            else:
                self.shadow[name] = param.detach().clone()

        valid_names = {name for name, _ in self._averaged_parameters(model)}
        for name in list(self.shadow.keys()):
            if name not in valid_names:
                del self.shadow[name]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1

        if self.num_updates <= self.update_after_steps:
            self.copy_from(model)
            return

        if (self.num_updates - self.update_after_steps) % self.update_every != 0:
            return

        decay = self.decay
        for name, param in self._averaged_parameters(model):
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return OrderedDict((name, tensor.detach()) for name, tensor in self.shadow.items())

    @torch.no_grad()
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        missing = []
        unexpected = [name for name in state_dict.keys() if name not in self.shadow]

        for name, shadow_tensor in self.shadow.items():
            if name not in state_dict:
                missing.append(name)
                continue
            shadow_tensor.copy_(state_dict[name].to(device=shadow_tensor.device, dtype=shadow_tensor.dtype))

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading EMA state_dict: missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    @torch.no_grad()
    def _swap_parameters(self, model: nn.Module) -> None:
        for name, param in self._averaged_parameters(model):
            if name not in self.shadow:
                continue
            raw_data = param.data
            shadow_data = self.shadow[name]
            if shadow_data.device != raw_data.device or shadow_data.dtype != raw_data.dtype:
                shadow_data = shadow_data.to(device=raw_data.device, dtype=raw_data.dtype)
            param.data = shadow_data
            self.shadow[name] = raw_data

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module):
        self._swap_parameters(model)
        try:
            yield
        finally:
            self._swap_parameters(model)
