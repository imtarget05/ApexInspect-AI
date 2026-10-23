"""Collate functions for different model formats."""

import torch
import numpy as np


def collate_fn(batch: list) -> dict:
    """Default collate function."""
    return torch.utils.data.default_collate(batch)


def yolov8_collate_fn(batch: list) -> dict:
    """
    Collate function for YOLOv8 training.
    
    Returns:
        dict with 'images' (B, 3, H, W) and 'labels' (N, 5) where 
        labels = [batch_idx, class_id, x_center, y_center, width, height]
    """
    images = torch.stack([item["image"] for item in batch])
    labels = []
    for i, item in enumerate(batch):
        lbl = item["labels"]
        if len(lbl) > 0:
            lbl = lbl.clone()
            # Add batch index as first column
            batch_idx = torch.full((len(lbl), 1), i, dtype=lbl.dtype)
            lbl = torch.cat([batch_idx, lbl], dim=1)
            labels.append(lbl)
    labels = torch.cat(labels, dim=0) if labels else torch.zeros((0, 6))
    return {"images": images, "labels": labels}


def detection_collate_fn(batch: list) -> dict:
    """Collate for generic detection (list of dicts)."""
    images = torch.stack([item["image"] for item in batch])
    targets = [{"boxes": item["labels"][:, 1:], "labels": item["labels"][:, 0].long()} for item in batch]
    return {"images": images, "targets": targets}


def classification_collate_fn(batch: list) -> dict:
    """Collate for classification."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"images": images, "labels": labels}


__all__ = [
    "collate_fn",
    "yolov8_collate_fn",
    "detection_collate_fn",
    "classification_collate_fn",
]