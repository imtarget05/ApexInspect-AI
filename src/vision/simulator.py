import random
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

class PCBCameraSimulator:
    """
    Generates synthetic industrial PCB inspection frames with optional injected defects.
    Provides realistic imagery for Edge Vision testing, ONNX inference benchmarking,
    and Streamlit live demo streams without requiring physical factory hardware.
    """

    DEFECT_CLASSES = ["short_circuit", "missing_hole", "mouse_bite", "open_circuit", "spur"]

    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height

    def generate_pcb_frame(self, inject_defect: bool = False, specific_defect: str = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Renders a synthetic PCB frame.
        Returns:
            frame: np.ndarray (BGR image)
            defects: List of dicts with keys: class, bbox [x1, y1, x2, y2], confidence
        """
        # 1. Base PCB Dark Green Solder Mask
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (20, 65, 25) # Deep industrial PCB green

        # Add subtle surface grain texture
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 2. Draw Copper Circuit Traces (Gold/Bronze color)
        copper_color = (40, 165, 215) # BGR
        pad_color = (200, 210, 215)   # Silver solder pads

        # Draw a grid of circuit tracks
        for y in range(80, self.height - 80, 50):
            cv2.line(img, (60, y), (self.width - 60, y), copper_color, 2)
        for x in range(100, self.width - 100, 70):
            cv2.line(img, (x, 60), (x, self.height - 60), copper_color, 2)

        # 3. Draw IC Footprints & Solder Pads
        ic_center_x, ic_center_y = self.width // 2, self.height // 2
        # IC chip body (matte black)
        cv2.rectangle(img, (ic_center_x - 70, ic_center_y - 70), (ic_center_x + 70, ic_center_y + 70), (30, 30, 30), -1)
        cv2.putText(img, "ARM-CORTEX", (ic_center_x - 55, ic_center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # IC Pins on four sides
        pin_boxes = []
        for offset in range(-50, 55, 15):
            # Left & Right pins
            cv2.rectangle(img, (ic_center_x - 90, ic_center_y + offset - 4), (ic_center_x - 70, ic_center_y + offset + 4), pad_color, -1)
            cv2.rectangle(img, (ic_center_x + 70, ic_center_y + offset - 4), (ic_center_x + 90, ic_center_y + offset + 4), pad_color, -1)
            pin_boxes.append((ic_center_x - 90, ic_center_y + offset - 4, ic_center_x - 70, ic_center_y + offset + 4))

        # 4. Draw Via Holes (through-hole vias)
        via_centers = []
        for x in [120, 200, 440, 520]:
            for y in [130, 250, 370, 490]:
                cv2.circle(img, (x, y), 8, copper_color, 2)
                cv2.circle(img, (x, y), 4, (10, 10, 10), -1) # Drill hole
                via_centers.append((x, y))

        # 5. Silkscreen Labels
        cv2.putText(img, "ApexInspect SMT-LINE-01", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 240, 220), 2)
        cv2.putText(img, "REV 3.2", (self.width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 180), 1)

        defects = []
        if inject_defect:
            defect_type = specific_defect if specific_defect in self.DEFECT_CLASSES else random.choice(self.DEFECT_CLASSES)

            if defect_type == "short_circuit":
                # Inject a solder bridge connecting two adjacent IC pins
                target_pin = pin_boxes[random.randint(0, len(pin_boxes) - 2)]
                x1, y1, x2, y2 = target_pin
                # Solder blob bridging pins
                cv2.circle(img, (x1 + 10, y1 + 10), 10, pad_color, -1)
                bbox = [max(0, x1 - 5), max(0, y1 - 2), min(self.width, x2 + 5), min(self.height, y1 + 22)]
                defects.append({
                    "class": "short_circuit",
                    "bbox": bbox,
                    "confidence": round(random.uniform(0.85, 0.98), 2)
                })

            elif defect_type == "missing_hole":
                # Via hole is solid copper without the drilled center hole
                via_x, via_y = random.choice(via_centers)
                cv2.circle(img, (via_x, via_y), 8, copper_color, -1)
                bbox = [via_x - 12, via_y - 12, via_x + 12, via_y + 12]
                defects.append({
                    "class": "missing_hole",
                    "bbox": bbox,
                    "confidence": round(random.uniform(0.82, 0.95), 2)
                })

            elif defect_type == "mouse_bite":
                # A copper track has a bitten-out notch
                track_y = 130
                cv2.circle(img, (250, track_y), 6, (20, 65, 25), -1) # Bite notch
                bbox = [240, track_y - 8, 260, track_y + 8]
                defects.append({
                    "class": "mouse_bite",
                    "bbox": bbox,
                    "confidence": round(random.uniform(0.80, 0.93), 2)
                })

            elif defect_type == "open_circuit":
                # A severed copper track
                track_y = 180
                cv2.rectangle(img, (310, track_y - 3), (330, track_y + 3), (20, 65, 25), -1)
                bbox = [305, track_y - 6, 335, track_y + 6]
                defects.append({
                    "class": "open_circuit",
                    "bbox": bbox,
                    "confidence": round(random.uniform(0.88, 0.97), 2)
                })

            else: # spur
                track_y = 230
                cv2.line(img, (280, track_y), (295, track_y + 12), copper_color, 2)
                bbox = [275, track_y - 2, 300, track_y + 15]
                defects.append({
                    "class": "spur",
                    "bbox": bbox,
                    "confidence": round(random.uniform(0.79, 0.91), 2)
                })

        return img, defects
