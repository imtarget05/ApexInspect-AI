import os
import json
import random
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

class PCBCameraSimulator:
    """
    Simulates high-speed edge camera stream on factory SMT assembly lines.
    Loads and streams real high-resolution photographic PCB circuit board images
    from industrial datasets (akhatova/pcb-defects), supporting both defect-free (PASS)
    and authentic manufacturing defect samples.
    """

    DEFECT_CLASSES = ["short_circuit", "missing_hole", "mouse_bite", "open_circuit", "spur", "spurious_copper"]

    def __init__(self, width: int = 640, height: int = 640, sample_dir: str = None):
        self.width = width
        self.height = height
        if sample_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            sample_dir = str(project_root / "data" / "sample_pcbs")
        self.sample_dir = Path(sample_dir)
        self.samples: Dict[str, Any] = {}
        self._load_samples()

    def _load_samples(self):
        """Loads curated real PCB images and their defect metadata."""
        meta_path = self.sample_dir / "sample_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)
            except Exception as e:
                print(f"[Simulator] Failed to load sample metadata ({e})")

    def generate_pcb_frame(self, inject_defect: bool = False, specific_defect: str = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Returns a real PCB frame (from actual industrial dataset).
        Returns:
            frame: np.ndarray (BGR image, 640x640)
            defects: List of dicts with keys: class, bbox [x1, y1, x2, y2], confidence
        """
        # 1. Try loading real images from curated sample directory
        if self.samples and self.sample_dir.exists():
            if not inject_defect:
                pass_keys = [k for k, v in self.samples.items() if v.get("class") == "pass"]
                if pass_keys:
                    chosen_key = random.choice(pass_keys)
                    img_p = self.sample_dir / chosen_key
                    im = cv2.imread(str(img_p))
                    if im is not None:
                        if im.shape[:2] != (self.height, self.width):
                            im = cv2.resize(im, (self.width, self.height))
                        return im, []
            else:
                target_cls = specific_defect if specific_defect in self.DEFECT_CLASSES else random.choice(self.DEFECT_CLASSES)
                defect_keys = [k for k, v in self.samples.items() if v.get("class") == target_cls]
                if not defect_keys and target_cls == "short":
                    defect_keys = [k for k, v in self.samples.items() if v.get("class") == "short_circuit"]
                if defect_keys:
                    chosen_key = random.choice(defect_keys)
                    img_p = self.sample_dir / chosen_key
                    im = cv2.imread(str(img_p))
                    if im is not None:
                        if im.shape[:2] != (self.height, self.width):
                            im = cv2.resize(im, (self.width, self.height))
                        raw_defects = self.samples[chosen_key].get("defects", [])
                        return im, [dict(d) for d in raw_defects]

        # 2. Fallback to procedural generation if real dataset images are missing
        return self._generate_procedural_frame(inject_defect, specific_defect)

    def _generate_procedural_frame(self, inject_defect: bool = False, specific_defect: str = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Procedural fallback frame generator."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (35, 85, 40)
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = cv2.add(img, np.full(img.shape, 30, dtype=np.uint8))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        copper_color = (40, 165, 215)
        pad_color = (200, 210, 215)

        for y in range(80, self.height - 80, 50):
            cv2.line(img, (60, y), (self.width - 60, y), copper_color, 2)
        for x in range(100, self.width - 100, 70):
            cv2.line(img, (x, 60), (x, self.height - 60), copper_color, 2)

        ic_center_x, ic_center_y = self.width // 2, self.height // 2
        cv2.rectangle(img, (ic_center_x - 70, ic_center_y - 70), (ic_center_x + 70, ic_center_y + 70), (30, 30, 30), -1)
        cv2.putText(img, "ARM-CORTEX", (ic_center_x - 55, ic_center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        pin_boxes = []
        for offset in range(-50, 55, 15):
            cv2.rectangle(img, (ic_center_x - 90, ic_center_y + offset - 4), (ic_center_x - 70, ic_center_y + offset + 4), pad_color, -1)
            cv2.rectangle(img, (ic_center_x + 70, ic_center_y + offset - 4), (ic_center_x + 90, ic_center_y + offset + 4), pad_color, -1)
            pin_boxes.append((ic_center_x - 90, ic_center_y + offset - 4, ic_center_x - 70, ic_center_y + offset + 4))

        via_centers = []
        for x in [120, 200, 440, 520]:
            for y in [130, 250, 370, 490]:
                cv2.circle(img, (x, y), 8, copper_color, 2)
                cv2.circle(img, (x, y), 4, (10, 10, 10), -1)
                via_centers.append((x, y))

        cv2.putText(img, "ApexInspect SMT-LINE-01", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 240, 220), 2)
        cv2.putText(img, "REV 3.2", (self.width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 180), 1)

        defects = []
        if inject_defect:
            defect_type = specific_defect if specific_defect in self.DEFECT_CLASSES else random.choice(self.DEFECT_CLASSES)
            if defect_type in ["short_circuit", "short"]:
                target_pin = pin_boxes[random.randint(0, len(pin_boxes) - 2)]
                x1, y1, x2, y2 = target_pin
                cv2.circle(img, (x1 + 10, y1 + 10), 10, pad_color, -1)
                bbox = [max(0, x1 - 5), max(0, y1 - 2), min(self.width, x2 + 5), min(self.height, y1 + 22)]
                defects.append({"class": "short_circuit", "bbox": bbox, "confidence": round(random.uniform(0.85, 0.98), 2)})
            elif defect_type == "missing_hole":
                via_x, via_y = random.choice(via_centers)
                cv2.circle(img, (via_x, via_y), 8, copper_color, -1)
                bbox = [via_x - 12, via_y - 12, via_x + 12, via_y + 12]
                defects.append({"class": "missing_hole", "bbox": bbox, "confidence": round(random.uniform(0.82, 0.95), 2)})
            elif defect_type == "mouse_bite":
                track_y = 130
                cv2.circle(img, (250, track_y), 6, (20, 65, 25), -1)
                bbox = [240, track_y - 8, 260, track_y + 8]
                defects.append({"class": "mouse_bite", "bbox": bbox, "confidence": round(random.uniform(0.80, 0.93), 2)})
            elif defect_type == "open_circuit":
                track_y = 180
                cv2.rectangle(img, (310, track_y - 3), (330, track_y + 3), (20, 65, 25), -1)
                bbox = [305, track_y - 6, 335, track_y + 6]
                defects.append({"class": "open_circuit", "bbox": bbox, "confidence": round(random.uniform(0.88, 0.97), 2)})
            else:
                track_y = 230
                cv2.line(img, (280, track_y), (295, track_y + 12), copper_color, 2)
                bbox = [275, track_y - 2, 300, track_y + 15]
                defects.append({"class": "spur", "bbox": bbox, "confidence": round(random.uniform(0.79, 0.91), 2)})

        return img, defects
