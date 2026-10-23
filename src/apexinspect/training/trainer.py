"""
Training loop for defect detection with callbacks, logging, and distributed support.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from apexinspect.models import DefectDetector, load_weights, save_weights
from apexinspect.metrics import compute_map, compute_f1_per_class
from apexinspect.config import Config


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_map: float = 0.0
    best_f1: float = 0.0
    metrics_history: List[Dict] = field(default_factory=list)


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: "DefectTrainer"): pass
    def on_train_end(self, trainer: "DefectTrainer"): pass
    def on_epoch_begin(self, trainer: "DefectTrainer", epoch: int): pass
    def on_epoch_end(self, trainer: "DefectTrainer", epoch: int, metrics: Dict): pass
    def on_batch_begin(self, trainer: "DefectTrainer", batch_idx: int): pass
    def on_batch_end(self, trainer: "DefectTrainer", batch_idx: int, loss: float): pass
    def on_validation_begin(self, trainer: "DefectTrainer"): pass
    def on_validation_end(self, trainer: "DefectTrainer", metrics: Dict): pass


class EarlyStopping(Callback):
    """Early stopping based on metric."""

    def __init__(self, monitor: str = "mAP50-95", patience: int = 20, mode: str = "max", min_delta: float = 1e-4):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.stopped = False

    def on_validation_end(self, trainer: "DefectTrainer", metrics: Dict):
        value = metrics.get(self.monitor, 0)
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
                print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")


class ModelCheckpoint(Callback):
    """Save model checkpoints."""

    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "epoch={epoch}-map={mAP50-95:.4f}.pt",
        monitor: str = "mAP50-95",
        mode: str = "max",
        save_best_only: bool = True,
        save_last: bool = True,
        every_n_epochs: int = 1,
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def on_validation_end(self, trainer: "DefectTrainer", metrics: Dict):
        epoch = trainer.state.epoch
        value = metrics.get(self.monitor, 0)

        # Save best
        is_best = False
        if self.mode == "max" and value > self.best_value:
            is_best = True
            self.best_value = value
        elif self.mode == "min" and value < self.best_value:
            is_best = True
            self.best_value = value

        if is_best and self.save_best_only:
            path = self.dirpath / f"best_{self.filename.format(epoch=epoch, **metrics)}"
            save_weights(
                trainer.model.model.model,
                path,
                epoch=epoch,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                metrics=metrics,
            )
            print(f"\nSaved best model to {path}")

        # Save last
        if self.save_last and epoch % self.every_n_epochs == 0:
            path = self.dirpath / f"last_{self.filename.format(epoch=epoch, **metrics)}"
            save_weights(
                trainer.model.model.model,
                path,
                epoch=epoch,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                metrics=metrics,
            )


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, trainer: "DefectTrainer", epoch: int, metrics: Dict):
        if hasattr(self.scheduler, "step"):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics.get("mAP50-95", 0))
            else:
                self.scheduler.step()

    def on_batch_end(self, trainer: "DefectTrainer", batch_idx: int, loss: float):
        # For batch-level schedulers (e.g., OneCycleLR)
        if hasattr(self.scheduler, "step") and not isinstance(
            self.scheduler, 
            (torch.optim.lr_scheduler.ReduceLROnPlateau, 
             torch.optim.lr_scheduler.CosineAnnealingLR,
             torch.optim.lr_scheduler.StepLR)
        ):
            self.scheduler.step()


class MLflowLogger(Callback):
    """MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "apexinspect",
        tracking_uri: str = "http://localhost:5000",
        tags: Optional[Dict] = None,
    ):
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_tracking_uri(tracking_uri)
            self.mlflow.set_experiment(experiment_name)
            self.run = None
            self.tags = tags or {}
        except ImportError:
            self.mlflow = None
            print("MLflow not installed, skipping MLflow logging")

    def on_train_begin(self, trainer: "DefectTrainer"):
        if self.mlflow:
            self.run = self.mlflow.start_run(tags=self.tags)
            # Log config
            self.mlflow.log_params(trainer.config.__dict__ if hasattr(trainer.config, "__dict__") else {})

    def on_epoch_end(self, trainer: "DefectTrainer", epoch: int, metrics: Dict):
        if self.mlflow and self.run:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, v, step=epoch)
            
            # Log learning rate
            if trainer.optimizer:
                lr = trainer.optimizer.param_groups[0]["lr"]
                self.mlflow.log_metric("lr", lr, step=epoch)

    def on_train_end(self, trainer: "DefectTrainer"):
        if self.mlflow and self.run:
            self.mlflow.end_run()


class WandbLogger(Callback):
    """Weights & Biases logger."""

    def __init__(
        self,
        project: str = "apexinspect",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict] = None,
    ):
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=project, entity=entity, tags=tags, config=config)
        except ImportError:
            self.wandb = None
            print("wandb not installed, skipping W&B logging")

    def on_epoch_end(self, trainer: "DefectTrainer", epoch: int, metrics: Dict):
        if self.wandb:
            log_dict = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if trainer.optimizer:
                log_dict["lr"] = trainer.optimizer.param_groups[0]["lr"]
            self.wandb.log(log_dict, step=epoch)

    def on_train_end(self, trainer: "DefectTrainer"):
        if self.wandb:
            self.wandb.finish()


class TensorBoardLogger(Callback):
    """TensorBoard logger."""

    def __init__(self, log_dir: str = "runs/apexinspect"):
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, trainer: "DefectTrainer", epoch: int, metrics: Dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, epoch)

        if trainer.optimizer:
            self.writer.add_scalar("lr", trainer.optimizer.param_groups[0]["lr"], epoch)

    def on_train_end(self, trainer: "DefectTrainer"):
        self.writer.close()


class ProgressLogger(Callback):
    """Console progress logger with rich formatting."""

    def __init__(self, log_every_n_batches: int = 50):
        self.log_every_n_batches = log_every_n_batches
        self.epoch_start_time = None

    def on_epoch_begin(self, trainer: "DefectTrainer", epoch: int):
        self.epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{trainer.config.training.epochs}")
        print(f"{'='*60}")

    def on_batch_end(self, trainer: "DefectTrainer", batch_idx: int, loss: float):
        if batch_idx % self.log_every_n_batches == 0:
            elapsed = time.time() - self.epoch_start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            eta = (len(trainer.train_loader) - batch_idx - 1) / batches_per_sec
            print(f"  Batch {batch_idx+1}/{len(trainer.train_loader)} | "
                  f"Loss: {loss:.4f} | "
                  f"{batches_per_sec:.1f} batches/s | "
                  f"ETA: {eta:.0f}s")

    def on_epoch_end(self, trainer: "DefectTrainer", epoch: int, metrics: Dict):
        elapsed = time.time() - self.epoch_start_time
        print(f"\nEpoch {epoch + 1} completed in {elapsed:.1f}s")
        print(f"  Train Loss: {metrics.get('train_loss', 0):.4f}")
        print(f"  Val mAP50-95: {metrics.get('mAP50-95', 0):.4f}")
        print(f"  Val mAP50: {metrics.get('mAP50', 0):.4f}")
        for cls, f1_info in metrics.get("F1_per_class", {}).items():
            if isinstance(f1_info, dict):
                print(f"  {cls}: P={f1_info.get('precision', 0):.3f} "
                      f"R={f1_info.get('recall', 0):.3f} "
                      f"F1={f1_info.get('f1', 0):.3f}")


class DefectTrainer:
    """
    Main trainer class for defect detection.
    Handles training loop, validation, callbacks, logging, and checkpointing.
    """

    def __init__(
        self,
        model: DefectDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[List[Callback]] = None,
        device: str = "auto",
        output_dir: str = "outputs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)

        # Distributed training setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        # State
        self.state = TrainingState()
        self.stop_training = False

        # Setup callbacks
        self._setup_default_callbacks()

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

    def _setup_default_callbacks(self):
        """Setup default callbacks if not provided."""
        default_callbacks = [
            ProgressLogger(),
            ModelCheckpoint(dirpath=self.output_dir / "checkpoints"),
            EarlyStopping(patience=self.config.training.patience),
        ]

        if self.config.logging.mlflow_uri:
            default_callbacks.append(MLflowLogger(
                experiment_name=self.config.experiment_name,
                tracking_uri=self.config.logging.mlflow_uri,
            ))

        if self.config.logging.wandb_project:
            default_callbacks.append(WandbLogger(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
            ))

        default_callbacks.append(TensorBoardLogger(log_dir=str(self.output_dir / "tensorboard")))

        # Add user callbacks
        self.callbacks = default_callbacks + self.callbacks

    def train(self) -> Dict:
        """Main training loop."""
        # Call on_train_begin callbacks
        for cb in self.callbacks:
            cb.on_train_begin(self)

        print(f"Training on {self.device} | Rank {self.rank}/{self.world_size}")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(self.state.epoch, self.config.training.epochs):
            if self.stop_training:
                break

            self.state.epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch(epoch)

            # Validation
            val_metrics = self.validate()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            self.state.metrics_history.append(metrics)

            # Call callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrics)

        # Call on_train_end callbacks
        for cb in self.callbacks:
            cb.on_train_end(self)

        return self.state.metrics_history[-1] if self.state.metrics_history else {}

    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for cb in self.callbacks:
            cb.on_epoch_begin(self, epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            for cb in self.callbacks:
                cb.on_batch_begin(self, batch_idx)

            loss = self._train_step(batch)

            total_loss += loss

            for cb in self.callbacks:
                cb.on_batch_end(self, batch_idx, loss)

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def _train_step(self, batch: Dict) -> float:
        """Single training step."""
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        if self.scaler:
            with torch.amp.autocast("cuda"):
                loss_dict = self.model.train_step({"images": images, "labels": labels})
                loss = loss_dict.get("loss", sum(loss_dict.values()))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict = self.model.train_step({"images": images, "labels": labels})
            loss = loss_dict.get("loss", sum(loss_dict.values()))
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def validate(self) -> Dict:
        """Run validation."""
        for cb in self.callbacks:
            cb.on_validation_begin(self)

        self.model.eval()
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(images)

                # Post-process predictions (YOLO-specific)
                preds = self._postprocess(outputs)
                gts = self._labels_to_detections(labels)

                all_preds.extend(preds)
                all_gts.extend(gts)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_gts)

        for cb in self.callbacks:
            cb.on_validation_end(self, metrics)

        return metrics

    def _postprocess(self, outputs) -> List[Dict]:
        """Convert model outputs to detection format."""
        # This depends on model architecture
        # For YOLOv8, use model.postprocess
        return self.model.postprocess(outputs) if hasattr(self.model, "postprocess") else []

    def _labels_to_detections(self, labels: torch.Tensor) -> List[Dict]:
        """Convert YOLO labels to detection format."""
        detections = []
        for batch_idx in range(int(labels[:, 0].max().item()) + 1):
            mask = labels[:, 0] == batch_idx
            batch_labels = labels[mask]
            if len(batch_labels) == 0:
                detections.append({"boxes": [], "scores": [], "class_ids": []})
            else:
                boxes = batch_labels[:, 2:].clone()
                # xywh normalized -> xyxy
                boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                detections.append({
                    "boxes": boxes.cpu().numpy(),
                    "scores": torch.ones(len(batch_labels)).cpu().numpy(),
                    "class_ids": batch_labels[:, 1].cpu().numpy().astype(int),
                })
        return detections

    def _compute_metrics(self, preds: List[Dict], gts: List[Dict]) -> Dict:
        """Compute validation metrics."""
        class_names = self.config.data.class_names
        
        # mAP
        map_results = compute_map(preds, gts, class_names=class_names)
        
        # F1 per class
        f1_results = compute_f1_per_class(preds, gts, class_names=class_names)
        
        return {
            **map_results,
            "F1_per_class": f1_results,
        }

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint."""
        save_weights(
            self.model.model.model,
            path,
            epoch=self.state.epoch,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=self.state.metrics_history[-1] if self.state.metrics_history else {},
        )

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.model.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state.epoch = checkpoint["epoch"]
        self.state.metrics_history = checkpoint.get("metrics_history", [])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0,
    log_every: int = 50,
) -> Dict:
    """Train for one epoch (standalone function)."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.amp.autocast("cuda"):
                loss_dict = loss_fn(model(images), labels)
                loss = loss_dict.get("loss", sum(loss_dict.values()))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = loss_fn(model(images), labels)
            loss = loss_dict.get("loss", sum(loss_dict.values()))
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_every == 0:
            print(f"Epoch {epoch} Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")

    return {"train_loss": total_loss / len(loader)}


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    compute_metrics: bool = True,
) -> Dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(images)
            loss_dict = loss_fn(outputs, labels)
            loss = loss_dict.get("loss", sum(loss_dict.values()))
            total_loss += loss.item()

            if compute_metrics:
                # Post-process (depends on model)
                pass

    return {"val_loss": total_loss / len(loader)}


__all__ = [
    "DefectTrainer",
    "train_one_epoch",
    "validate",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "MLflowLogger",
    "WandbLogger",
    "TensorBoardLogger",
    "ProgressLogger",
]