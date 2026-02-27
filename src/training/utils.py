import math
import yaml
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(
        self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]

        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )

            factor = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience=10, mode="min"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.should_stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return True

        improved = (self.mode == "min" and metric < self.best) or (
            self.mode == "max" and metric > self.best
        )

        if improved:
            self.best = metric
            self.counter = 0
            return True
            
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class AverageMeter:
    """Tracks running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(path):
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(state, path):
    """Save a training checkpoint."""
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
