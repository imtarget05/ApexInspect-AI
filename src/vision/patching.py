import time
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

from .detector import PCBDefectDetector


class HighResPatchInferencer:
    """
    Sliced Automated Hyper Inference (SAHI-style) for High-Resolution PCB Inspection.
    Slices ultra-high-resolution images (e.g. 2048x2048, 4096x4096) into overlapping patches,
    runs YOLOv8 micro-inspection on each patch, and merges detections globally using NMS.
    Prevents microscopic defects (short circuits, mouse bites) from disappearing during resizing.
    """

    def __init__(
        self,
        detector: Optional[PCBDefectDetector] = None,
        patch_size: Tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.20,
        iou_threshold: float = 0.45
    ):
        self.detector = detector or PCBDefectDetector()
        self.patch_w, self.patch_h = patch_size
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold

    def generate_slices(self, img_h: int, img_w: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculates bounding boxes (x1, y1, x2, y2) for overlapping tiles covering the image.
        """
        # If image is smaller than or equal to patch size, return single slice
        if img_w <= self.patch_w and img_h <= self.patch_h:
            return [(0, 0, img_w, img_h)]

        step_x = int(round(self.patch_w * (1.0 - self.overlap_ratio)))
        step_y = int(round(self.patch_h * (1.0 - self.overlap_ratio)))
        step_x = max(step_x, 32)
        step_y = max(step_y, 32)

        x_starts = list(range(0, max(1, img_w - self.patch_w + 1), step_x))
        if not x_starts or x_starts[-1] + self.patch_w < img_w:
            x_starts.append(max(0, img_w - self.patch_w))

        y_starts = list(range(0, max(1, img_h - self.patch_h + 1), step_y))
        if not y_starts or y_starts[-1] + self.patch_h < img_h:
            y_starts.append(max(0, img_h - self.patch_h))

        # Deduplicate and sort starts
        x_starts = sorted(list(set(x_starts)))
        y_starts = sorted(list(set(y_starts)))

        slices: List[Tuple[int, int, int, int]] = []
        for ys in y_starts:
            for xs in x_starts:
                x1 = xs
                y1 = ys
                x2 = min(xs + self.patch_w, img_w)
                y2 = min(ys + self.patch_h, img_h)
                slices.append((x1, y1, x2, y2))

        return slices

    def apply_global_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies Non-Maximum Suppression (NMS) across all patches to deduplicate detections.
        """
        if not detections:
            return []

        boxes = []
        scores = []
        classes = []

        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            boxes.append([int(x1), int(y1), int(w), int(h)])
            scores.append(float(d.get("confidence", 0.5)))
            classes.append(d.get("class", "defect"))

        # Perform class-wise NMS to avoid suppressing different defect types that coincide
        merged_detections: List[Dict[str, Any]] = []
        unique_classes = set(classes)

        for target_cls in unique_classes:
            cls_indices = [i for i, c in enumerate(classes) if c == target_cls]
            cls_boxes = [boxes[i] for i in cls_indices]
            cls_scores = [scores[i] for i in cls_indices]

            nms_indices = cv2.dnn.NMSBoxes(
                cls_boxes,
                cls_scores,
                self.detector.conf_threshold,
                self.iou_threshold
            )

            if len(nms_indices) > 0:
                for idx in nms_indices.flatten():
                    original_idx = cls_indices[idx]
                    b = boxes[original_idx]
                    merged_detections.append({
                        "class": target_cls,
                        "confidence": round(scores[original_idx], 2),
                        "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                    })

        return merged_detections

    def infer_high_res(
        self,
        img: np.ndarray,
        include_full_image: bool = True,
        ground_truth: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Performs high-resolution patched inference.
        Returns:
            annotated_frame: High-res annotated image with global bounding boxes.
            detections: Merged list of defect detections in full image coordinates.
            latency_ms: Total inference time across all tiles in milliseconds.
        """
        start_time = time.perf_counter()
        img_h, img_w = img.shape[:2]

        all_detections: List[Dict[str, Any]] = []

        # 1. Global context pass (resized full image) if requested
        if include_full_image:
            _, global_dets, _ = self.detector.infer(img, ground_truth_defects=ground_truth)
            all_detections.extend(global_dets)

        # 2. Sliced patch inference
        slices = self.generate_slices(img_h, img_w)
        for (x1, y1, x2, y2) in slices:
            patch = img[y1:y2, x1:x2]
            if patch.shape[0] < 16 or patch.shape[1] < 16:
                continue

            # Sub-ground truth within this patch for simulation/fallback
            patch_gt = []
            if ground_truth:
                for gt in ground_truth:
                    gx1, gy1, gx2, gy2 = gt.get("bbox", [0, 0, 0, 0])
                    # Check if bounding box intersects patch
                    ix1 = max(x1, gx1)
                    iy1 = max(y1, gy1)
                    ix2 = min(x2, gx2)
                    iy2 = min(y2, gy2)
                    if ix2 > ix1 and iy2 > iy1:
                        # Local coords inside patch
                        patch_gt.append({
                            "class": gt.get("class", "defect"),
                            "confidence": gt.get("confidence", 0.90),
                            "bbox": [ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1]
                        })

            _, patch_dets, _ = self.detector.infer(patch, ground_truth_defects=patch_gt)

            # Offset patch coordinates to full image space
            for det in patch_dets:
                bx1, by1, bx2, by2 = det.get("bbox", [0, 0, 0, 0])
                all_detections.append({
                    "class": det.get("class", "defect"),
                    "confidence": det.get("confidence", 0.5),
                    "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                })

        # 3. Global Non-Maximum Suppression
        merged_detections = self.apply_global_nms(all_detections)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Render annotations on high-res frame
        annotated_frame = self.detector.render_annotations(img.copy(), merged_detections, total_latency_ms)

        return annotated_frame, merged_detections, total_latency_ms
