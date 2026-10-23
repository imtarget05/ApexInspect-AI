"""
Custom loss functions for defect detection.
Includes YOLOv8 detection loss with configurable gains.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from ultralytics.utils.loss import v8DetectionLoss as UltralyticsV8Loss


class DetectionLoss(nn.Module):
    """
    Generic detection loss wrapper.
    Combines box, class, and DFL losses with configurable gains.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        num_classes: int = 10,
        reg_max: int = 16,
        tal_topk: int = 10,
    ):
        super().__init__()
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain

        # Use ultralytics v8 loss
        self.loss_fn = UltralyticsV8Loss(model)
        self.loss_fn.box_gain = box_gain
        self.loss_fn.cls_gain = cls_gain
        self.loss_fn.dfl_gain = dfl_gain

    def forward(
        self,
        preds: Tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            preds: Model predictions (list of tensors from each head)
            targets: Ground truth labels [batch_idx, class, x, y, w, h]
            
        Returns:
            total_loss, loss_components (box, cls, dfl)
        """
        loss, loss_items = self.loss_fn(preds, targets)
        return loss, loss_items


class v8DetectionLoss(nn.Module):
    """
    YOLOv8 detection loss with custom gains.
    Direct wrapper around ultralytics loss for full control.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
    ):
        super().__init__()
        from ultralytics.utils.loss import v8DetectionLoss as UltralyticsLoss
        self.loss_fn = UltralyticsLoss(model)
        self.loss_fn.box_gain = box_gain
        self.loss_fn.cls_gain = cls_gain
        self.loss_fn.dfl_gain = dfl_gain

    def forward(self, preds, targets):
        return self.loss_fn(preds, targets)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BboxLoss(nn.Module):
    """IoU-based bounding box loss (CIoU, DIoU, GIoU)."""

    def __init__(self, iou_type: str = "ciou", reduction: str = "mean"):
        super().__init__()
        self.iou_type = iou_type
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss.
        
        Args:
            pred_boxes: Predicted boxes in xyxy format (N, 4)
            target_boxes: Target boxes in xyxy format (N, 4)
        """
        # Convert to xywh for IoU calculation
        pred_xywh = self._xyxy_to_xywh(pred_boxes)
        target_xywh = self._xyxy_to_xywh(target_boxes)

        iou = self._box_iou(pred_xywh, target_xywh)

        if self.iou_type == "iou":
            loss = 1 - iou
        elif self.iou_type == "giou":
            loss = 1 - self._giou(pred_xywh, target_xywh)
        elif self.iou_type == "diou":
            loss = 1 - self._diou(pred_xywh, target_xywh)
        elif self.iou_type == "ciou":
            loss = 1 - self._ciou(pred_xywh, target_xywh)
        else:
            raise ValueError(f"Unknown IoU type: {self.iou_type}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _xyxy_to_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes."""
        # boxes in xywh format
        x1, y1, w1, h1 = boxes1.unbind(-1)
        x2, y2, w2, h2 = boxes2.unbind(-1)

        # Intersection
        x1_i = torch.max(x1 - w1 / 2, x2 - w2 / 2)
        y1_i = torch.max(y1 - h1 / 2, y2 - h2 / 2)
        x2_i = torch.min(x1 + w1 / 2, x2 + w2 / 2)
        y2_i = torch.min(y1 + h1 / 2, y2 + h2 / 2)

        inter_w = (x2_i - x1_i).clamp(min=0)
        inter_h = (y2_i - y1_i).clamp(min=0)
        intersection = inter_w * inter_h

        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / (union + 1e-7)

    def _giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        iou = self._box_iou(boxes1, boxes2)

        # Enclosing box
        x1 = torch.min(boxes1[..., 0] - boxes1[..., 2] / 2, boxes2[..., 0] - boxes2[..., 2] / 2)
        y1 = torch.min(boxes1[..., 1] - boxes1[..., 3] / 2, boxes2[..., 1] - boxes2[..., 3] / 2)
        x2 = torch.max(boxes1[..., 0] + boxes1[..., 2] / 2, boxes2[..., 0] + boxes2[..., 2] / 2)
        y2 = torch.max(boxes1[..., 1] + boxes1[..., 3] / 2, boxes2[..., 1] + boxes2[..., 3] / 2)

        c_w = x2 - x1
        c_h = y2 - y1
        c_area = c_w * c_h + 1e-7

        area1 = boxes1[..., 2] * boxes1[..., 3]
        area2 = boxes2[..., 2] * boxes2[..., 3]
        union = area1 + area2 - iou * union

        giou = iou - (c_area - union) / c_area
        return giou

    def _diou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        iou = self._box_iou(boxes1, boxes2)

        # Center distance
        c2 = (boxes1[..., 0] - boxes2[..., 0]) ** 2 + (boxes1[..., 1] - boxes2[..., 1]) ** 2

        # Diagonal of enclosing box
        x1 = torch.min(boxes1[..., 0] - boxes1[..., 2] / 2, boxes2[..., 0] - boxes2[..., 2] / 2)
        y1 = torch.min(boxes1[..., 1] - boxes1[..., 3] / 2, boxes2[..., 1] - boxes2[..., 3] / 2)
        x2 = torch.max(boxes1[..., 0] + boxes1[..., 2] / 2, boxes2[..., 0] + boxes2[..., 2] / 2)
        y2 = torch.max(boxes1[..., 1] + boxes1[..., 3] / 2, boxes2[..., 1] + boxes2[..., 3] / 2)

        c_w = x2 - x1
        c_h = y2 - y1
        c_diag = c_w ** 2 + c_h ** 2 + 1e-7

        diou = iou - c2 / c_diag
        return diou

    def _ciou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        diou = self._diou(boxes1, boxes2)
        iou = self._box_iou(boxes1, boxes2)

        # Aspect ratio consistency
        w1, h1 = boxes1[..., 2], boxes1[..., 3]
        w2, h2 = boxes2[..., 2], boxes2[..., 3]

        v = (4 / (torch.pi ** 2)) * (torch.atan(w1 / (h1 + 1e-7)) - torch.atan(w2 / (h2 + 1e-7))) ** 2
        alpha = v / (1 - iou + v + 1e-7)

        ciou = diou - alpha * v
        return ciou


class DFLLoss(nn.Module):
    """Distribution Focal Loss for bounding box regression."""

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Distribution Focal Loss.
        
        Args:
            pred_dist: Predicted distribution (N, 4, reg_max+1)
            target: Target distances (N, 4) in [0, reg_max]
        """
        target_left = target.floor().long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()

        loss_left = nn.functional.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1),
            target_left.view(-1),
            reduction="none"
        ).view_as(target)

        loss_right = nn.functional.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1),
            target_right.view(-1),
            reduction="none"
        ).view_as(target)

        loss = (weight_left * loss_left + weight_right * loss_right).mean()
        return loss


def build_loss(
    model: torch.nn.Module,
    loss_type: str = "v8",
    box_gain: float = 7.5,
    cls_gain: float = 0.5,
    dfl_gain: float = 1.5,
) -> nn.Module:
    """Factory function to build loss."""
    if loss_type == "v8":
        return v8DetectionLoss(model, box_gain, cls_gain, dfl_gain)
    elif loss_type == "custom":
        return DetectionLoss(model, box_gain, cls_gain, dfl_gain)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    "DetectionLoss",
    "v8DetectionLoss",
    "FocalLoss",
    "BboxLoss",
    "DFLLoss",
    "build_loss",
]