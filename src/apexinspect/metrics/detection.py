"""
Detection metrics: mAP, AP per class, F1, confusion matrix.
Compatible with COCO evaluation and YOLO format.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from collections import defaultdict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-7)


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """Compute Average Precision from precision-recall curve."""
    # Add sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05),
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute mAP@0.5:0.95 (COCO style).
    
    Args:
        predictions: List of dicts with 'boxes' (N,4 xyxy), 'scores' (N,), 'class_ids' (N,)
        ground_truths: List of dicts with 'boxes' (M,4 xyxy), 'class_ids' (M,)
        iou_thresholds: IoU thresholds for mAP calculation
        class_names: Optional class names for per-class results
        
    Returns:
        Dict with 'mAP50', 'mAP50-95', 'AP_per_class', 'AP_per_class_per_iou'
    """
    if not predictions or not ground_truths:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "AP_per_class": {}, "AP_per_class_per_iou": {}}

    num_classes = max(
        max([p["class_ids"].max() for p in predictions if len(p["class_ids"]) > 0], default=0),
        max([g["class_ids"].max() for g in ground_truths if len(g["class_ids"]) > 0], default=0),
    ) + 1

    class_names = class_names or [str(i) for i in range(num_classes)]

    # Collect all predictions and GTs per class
    aps = {c: [] for c in range(num_classes)}

    for iou_thr in iou_thresholds:
        for class_id in range(num_classes):
            # Get predictions for this class
            pred_boxes = []
            pred_scores = []
            for p in predictions:
                mask = p["class_ids"] == class_id
                pred_boxes.extend(p["boxes"][mask])
                pred_scores.extend(p["scores"][mask])

            if not pred_boxes:
                aps[class_id].append(0.0)
                continue

            pred_boxes = np.array(pred_boxes)
            pred_scores = np.array(pred_scores)

            # Sort by confidence
            order = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

            # Get GT for this class
            gt_boxes = []
            for g in ground_truths:
                mask = g["class_ids"] == class_id
                gt_boxes.extend(g["boxes"][mask])

            if not gt_boxes:
                aps[class_id].append(0.0)
                continue

            gt_boxes = np.array(gt_boxes)
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            # Compute TP/FP
            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))

            for i, pbox in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                for j, gbox in enumerate(gt_boxes):
                    if gt_matched[j]:
                        continue
                    iou = compute_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_thr:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[i] = 1

            # Compute cumulative precision/recall
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / (len(gt_boxes) + 1e-7)
            precisions = cum_tp / (cum_tp + cum_fp + 1e-7)

            ap = compute_ap(recalls, precisions)
            aps[class_id].append(ap)

    # Average over IoU thresholds
    ap_per_class = {}
    for class_id in range(num_classes):
        if aps[class_id]:
            ap_per_class[class_names[class_id]] = float(np.mean(aps[class_id]))

    mAP50 = np.mean([aps[c][0] for c in range(num_classes) if aps[c]])
    mAP50_95 = np.mean([np.mean(aps[c]) for c in range(num_classes) if aps[c]])

    return {
        "mAP50": float(mAP50),
        "mAP50-95": float(mAP50_95),
        "AP_per_class": ap_per_class,
        "AP_per_class_per_iou": {class_names[c]: [float(a) for a in aps[c]] for c in range(num_classes) if aps[c]},
    }


def compute_ap_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute AP at single IoU threshold per class."""
    result = compute_map(predictions, ground_truths, np.array([iou_threshold]), class_names)
    return result["AP_per_class"]


def compute_f1_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 per class at given confidence threshold.
    """
    if not predictions or not ground_truths:
        return {}

    num_classes = max(
        max([p["class_ids"].max() for p in predictions if len(p["class_ids"]) > 0], default=0),
        max([g["class_ids"].max() for g in ground_truths if len(g["class_ids"]) > 0], default=0),
    ) + 1

    class_names = class_names or [str(i) for i in range(num_classes)]
    results = {}

    for class_id in range(num_classes):
        # Filter predictions by confidence
        pred_boxes = []
        pred_scores = []
        for p in predictions:
            mask = (p["class_ids"] == class_id) & (p["scores"] >= conf_threshold)
            pred_boxes.extend(p["boxes"][mask])
            pred_scores.extend(p["scores"][mask])

        if not pred_boxes:
            results[class_names[class_id]] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue

        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)

        # Get GT
        gt_boxes = []
        for g in ground_truths:
            mask = g["class_ids"] == class_id
            gt_boxes.extend(g["boxes"][mask])

        if not gt_boxes:
            results[class_names[class_id]] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue

        gt_boxes = np.array(gt_boxes)

        # Match predictions to GT
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        tp = 0
        fp = 0

        for pbox in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for j, gbox in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        fn = len(gt_boxes) - gt_matched.sum()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        results[class_names[class_id]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    return results


class ConfusionMatrix:
    """Confusion matrix for detection evaluation."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)  # +1 for background/FP

    def update(self, predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5):
        """Update confusion matrix with batch predictions."""
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_classes = pred["class_ids"]

            gt_boxes = gt["boxes"]
            gt_classes = gt["class_ids"]

            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue

            if len(pred_boxes) == 0:
                # All GT are false negatives
                for gc in gt_classes:
                    self.matrix[gc, self.num_classes] += 1  # background column
                continue

            if len(gt_boxes) == 0:
                # All predictions are false positives
                for pc in pred_classes:
                    self.matrix[self.num_classes, pc] += 1  # background row
                continue

            # Match predictions to GT
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            for i, (pbox, pcls) in enumerate(zip(pred_boxes, pred_classes)):
                best_iou = 0
                best_gt_idx = -1
                for j, (gbox, gcls) in enumerate(zip(gt_boxes, gt_classes)):
                    if gt_matched[j]:
                        continue
                    iou = compute_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold:
                    gt_matched[best_gt_idx] = True
                    gt_cls = gt_classes[best_gt_idx]
                    self.matrix[gt_cls, pcls] += 1  # TP or misclassification
                else:
                    self.matrix[self.num_classes, pcls] += 1  # FP

            # Unmatched GT are FN
            for j, matched in enumerate(gt_matched):
                if not matched:
                    self.matrix[gt_classes[j], self.num_classes] += 1

    def compute_metrics(self) -> Dict:
        """Compute per-class precision, recall, F1 from confusion matrix."""
        results = {}
        for i in range(self.num_classes):
            tp = self.matrix[i, i]
            fp = self.matrix[self.num_classes, i]
            fn = self.matrix[i, self.num_classes]

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)

            results[self.class_names[i]] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

        # Overall
        total_tp = np.trace(self.matrix[:self.num_classes, :self.num_classes])
        total_fp = self.matrix[self.num_classes, :self.num_classes].sum()
        total_fn = self.matrix[:self.num_classes, self.num_classes].sum()

        overall_precision = total_tp / (total_tp + total_fp + 1e-7)
        overall_recall = total_tp / (total_tp + total_fn + 1e-7)
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-7)

        results["overall"] = {
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1": float(overall_f1),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
        }

        return results

    def normalize(self) -> np.ndarray:
        """Return normalized confusion matrix (row-wise)."""
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        return self.matrix / (row_sums + 1e-7)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "matrix": self.matrix.tolist(),
            "class_names": self.class_names,
        }


def compute_confusion_matrix(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> ConfusionMatrix:
    """Compute confusion matrix from predictions and ground truths."""
    cm = ConfusionMatrix(len(class_names), class_names)
    cm.update(predictions, ground_truths, iou_threshold)
    return cm


__all__ = [
    "compute_map",
    "compute_ap_per_class",
    "compute_f1_per_class",
    "compute_confusion_matrix",
    "ConfusionMatrix",
]