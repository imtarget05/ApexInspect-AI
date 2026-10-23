"""
Learning rate schedulers with warmup support.
"""

import math
from typing import Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine annealing with warmup.
    
    LR increases linearly during warmup, then follows cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class OneCycleLR(_LRScheduler):
    """
    OneCycle learning rate scheduler.
    LR increases to max_lr then decreases following cosine annealing.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.step_up = int(pct_start * total_steps)
        self.step_down = total_steps - self.step_up
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.step_up:
            # Increasing phase
            progress = self.last_epoch / max(self.step_up, 1)
            return [
                base_lr / self.div_factor + 
                (self.max_lr - base_lr / self.div_factor) * progress
                for base_lr in self.base_lrs
            ]
        else:
            # Decreasing phase
            progress = (self.last_epoch - self.step_up) / max(self.step_down, 1)
            if self.anneal_strategy == "cos":
                factor = (1 + math.cos(math.pi * progress)) / 2
            else:  # linear
                factor = 1 - progress
            
            return [
                base_lr / self.div_factor + 
                (self.max_lr - base_lr / self.div_factor) * factor
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = min(self.last_epoch / max(self.total_steps, 1), 1.0)
        factor = (1 - progress) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


class WarmupScheduler(_LRScheduler):
    """Wrapper that adds warmup to any scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        after_scheduler: _LRScheduler,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

    def step(self, epoch=None):
        if self.finished:
            self.after_scheduler.step(epoch)
        else:
            super().step(epoch)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    max_epochs: int = 100,
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
    **kwargs,
) -> _LRScheduler:
    """
    Factory function to create scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: "cosine", "onecycle", "poly", "step", "plateau"
        max_epochs: Total epochs
        warmup_epochs: Warmup epochs
        min_lr: Minimum learning rate
    """
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr
        )
        if warmup_epochs > 0:
            return WarmupScheduler(optimizer, warmup_epochs, scheduler)
        return scheduler

    elif scheduler_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 0.01),
            total_steps=max_epochs,
            pct_start=warmup_epochs / max_epochs,
            **{k: v for k, v in kwargs.items() if k != "max_lr"},
        )

    elif scheduler_type == "poly":
        scheduler = PolynomialLR(optimizer, max_epochs - warmup_epochs, **kwargs)
        if warmup_epochs > 0:
            return WarmupScheduler(optimizer, warmup_epochs, scheduler)
        return scheduler

    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )

    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            min_lr=min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


__all__ = [
    "CosineAnnealingWarmup",
    "OneCycleLR",
    "PolynomialLR",
    "WarmupScheduler",
    "get_scheduler",
]