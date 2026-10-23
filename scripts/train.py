#!/usr/bin/env python3
"""
Training script for ApexInspect-AI defect detection.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apexinspect.config import load_config, Config
from apexinspect.data import create_dataloaders
from apexinspect.models import DefectDetector, create_model, v8DetectionLoss
from apexinspect.training import DefectTrainer, get_scheduler
from apexinspect.utils import set_seed, get_device, setup_logging


def setup_ddp():
    """Setup distributed data parallel."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train ApexInspect defect detection model")
    parser.add_argument("--config", type=Path, help="Path to config YAML")
    parser.add_argument("--data", type=Path, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n", help="Model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp", action="store_true", help="Use distributed training")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--mlflow", action="store_true", help="Use MLflow")
    
    args = parser.parse_args()
    
    # Setup DDP
    if args.ddp:
        rank, world_size, local_rank = setup_ddp()
        is_main = rank == 0
    else:
        rank, world_size, local_rank = 0, 1, 0
        is_main = True
    
    # Set seed
    set_seed(args.seed + rank)
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.ddp:
        device = f"cuda:{local_rank}"
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Create config from args
        config = Config()
        config.model.name = args.model
        config.training.epochs = args.epochs
        config.data.batch_size = args.batch_size
        config.model.img_size = args.img_size
        config.optimizer.lr = args.lr
        config.training.seed = args.seed
        config.output_dir = str(args.output_dir)
    
    # Override data path if provided
    if args.data:
        config.data.train_path = str(args.data.parent / "images/train")
        config.data.val_path = str(args.data.parent / "images/val")
        config.data.test_path = str(args.data.parent / "images/test")
    
    # Setup logging
    logger = setup_logging(
        log_file=args.output_dir / "training.log",
        level="INFO",
    )
    
    if is_main:
        logger.info("Starting training", config=config.__dict__ if hasattr(config, "__dict__") else str(config))
    
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=config.data.train_path.replace("/images/train", ""),
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            img_size=config.model.img_size,
            class_names=config.data.class_names,
            mosaic=True,
            mosaic_prob=1.0,
            mixup=True,
            mixup_prob=0.1,
            aug_config=config.augmentation.__dict__ if hasattr(config.augmentation, "__dict__") else {},
        )
        
        # Create model
        model = create_model(
            model_name=config.model.name,
            num_classes=config.model.num_classes,
            img_size=config.model.img_size,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            confidence_threshold=config.model.confidence_threshold,
            iou_threshold=config.model.iou_threshold,
            max_det=config.model.max_det,
        )
        
        if args.ddp:
            model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)
        else:
            model = model.to(device)
        
        # Optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )
        
        # Scheduler
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="cosine",
            max_epochs=config.training.epochs,
            warmup_epochs=config.training.warmup_epochs,
            min_lr=1e-6,
        )
        
        # Loss
        loss_fn = v8DetectionLoss(model)
        
        # Resume if specified
        start_epoch = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            if is_main:
                logger.info(f"Resumed from epoch {start_epoch}")
        
        # Trainer
        trainer = DefectTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            output_dir=args.output_dir,
        )
        
        # Train
        if is_main:
            logger.info(f"Training started for {config.training.epochs} epochs")
        
        metrics = trainer.train()
        
        if is_main:
            logger.info("Training completed", final_metrics=metrics)
    
    finally:
        if args.ddp:
            cleanup_ddp()


if __name__ == "__main__":
    main()