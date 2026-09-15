import os
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def resolve_model_path(provided: Optional[str] = None) -> str:
    """Resolve the canonical ONNX model path.

    Priority: explicit arg -> $MODEL_PATH -> models/yolov8n_pcb_defect.onnx
    -> notebooks/best.onnx (Colab export fallback).
    Relative paths are resolved against the project root.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    candidates: List[str] = []
    if provided:
        candidates.append(provided)
    env_path = os.getenv("MODEL_PATH", "").strip().strip("\"'")
    if env_path:
        candidates.append(env_path)
    candidates += [
        "models/yolov8n_pcb_defect.onnx",
        "notebooks/best.onnx",
    ]
    for cand in candidates:
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
            continue
        abs_cand = os.path.join(project_root, cand)
        if os.path.exists(abs_cand):
            return abs_cand
        if os.path.exists(cand):
            return os.path.abspath(cand)
    # Nothing found: return the canonical destination (absolute) so
    # error messages / get_model_info() point at the expected location.
    return os.path.join(project_root, "models/yolov8n_pcb_defect.onnx")


class PCBDefectDetector:
    """
    Industrial PCB Defect Detection Engine powered by ONNX Runtime and OpenCV.
    Supports CPU-optimized execution, YOLOv8 NMS postprocessing, and fallback simulation.
    """

    CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short_circuit", "spur", "spurious_copper"]

    # Raw training labels (see notebooks/data.yaml: `short`) mapped to the
    # canonical names used across the app (simulator, backend, SOP RAG).
    CLASS_ALIASES = {
        "short": "short_circuit",
    }

    COLOR_MAP = {
        "short_circuit": (0, 0, 255),    # Red
        "short": (0, 0, 255),            # Red
        "missing_hole": (0, 165, 255),   # Orange
        "mouse_bite": (0, 255, 255),     # Yellow
        "open_circuit": (255, 0, 255),   # Magenta
        "spur": (255, 140, 0),           # Deep Cyan/Blue
        "spurious_copper": (200, 100, 0)
    }

    def __init__(self, model_path: str = None, conf_threshold: float = None):
        resolved = resolve_model_path(model_path)
        if conf_threshold is None:
            try:
                conf_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
            except ValueError:
                conf_threshold = 0.50
        self.model_path = resolved
        self.conf_threshold = conf_threshold
        self.session = None
        self.use_onnx = False
        self.input_name = ""
        self.output_name = ""
        self.model_meta: Dict[str, Any] = {}

        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes ONNX Runtime session if the model file is available."""
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Use available CPU cores (cloud 2-vCPU target: 2 threads; local: more).
                try:
                    opts.intra_op_num_threads = max(2, (os.cpu_count() or 2) // 2)
                except Exception:
                    opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.use_onnx = True
                self.model_meta = {
                    "input_name": self.input_name,
                    "input_shape": [int(d) if isinstance(d, int) else str(d) for d in self.session.get_inputs()[0].shape],
                    "output_name": self.output_name,
                    "output_shape": [int(d) if isinstance(d, int) else str(d) for d in self.session.get_outputs()[0].shape],
                }
                print(f"[Detector] Successfully loaded ONNX model from {self.model_path}")
            except Exception as e:
                print(f"[Detector] Failed to load ONNX runtime ({e}). Running in simulation mode.")
                self.use_onnx = False
        else:
            print(f"[Detector] Model file not found at {self.model_path}. Running in simulation mode.")
            self.use_onnx = False

    def normalize_class(self, raw_name: str) -> str:
        """Maps raw training labels to canonical defect names (short -> short_circuit)."""
        return self.CLASS_ALIASES.get(raw_name, raw_name)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns deployment metadata for the loaded ONNX model."""
        return {
            "model_path": self.model_path,
            "exists": os.path.exists(self.model_path),
            "use_onnx": self.use_onnx,
            "conf_threshold": self.conf_threshold,
            "class_names": list(self.CLASS_NAMES),
            "class_aliases": dict(self.CLASS_ALIASES),
            **self.model_meta,
        }

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Applies OpenCV CLAHE contrast enhancement and normalization."""
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        blob = cv2.dnn.blobFromImage(enhanced, 1/255.0, (640, 640), swapRB=True, crop=False)
        return blob

    def _postprocess_yolov8(self, raw_output: np.ndarray, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        """
        Parses YOLOv8 ONNX raw output (1, 4+nc, 8400) and applies Non-Maximum Suppression (NMS).
        """
        try:
            # Shape: (1, 4 + num_classes, 8400) -> Transpose to (8400, 4 + num_classes)
            preds = np.squeeze(raw_output)
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T

            boxes = []
            confidences = []
            class_ids = []

            # Coordinate scaling factors
            x_factor = orig_w / 640.0
            y_factor = orig_h / 640.0

            scores_matrix = preds[:, 4:]
            max_class_ids = np.argmax(scores_matrix, axis=1)
            max_scores = np.max(scores_matrix, axis=1)

            # Filter candidates by confidence threshold
            valid_mask = max_scores >= self.conf_threshold
            valid_preds = preds[valid_mask]
            valid_scores = max_scores[valid_mask]
            valid_classes = max_class_ids[valid_mask]

            for pred, score, cls_id in zip(valid_preds, valid_scores, valid_classes):
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(float(score))
                class_ids.append(int(cls_id))

            # Apply Non-Maximum Suppression (NMS)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.45)
            detections = []
            if len(indices) > 0:
                for idx in indices.flatten():
                    b = boxes[idx]
                    cls_id = class_ids[idx]
                    raw_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"defect_{cls_id}"
                    cls_name = self.normalize_class(raw_name)
                    detections.append({
                        "class": cls_name,
                        "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                        "confidence": round(confidences[idx], 2)
                    })
            return detections
        except Exception as e:
            print(f"[Detector] Note during postprocessing: {e}")
            return []

    def infer(self, frame: np.ndarray, ground_truth_defects: List[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Runs defect detection on an input image.
        Returns:
            annotated_frame: Image with bounding boxes and HUD.
            detections: List of defect detections.
            latency_ms: Inference time in milliseconds.
        """
        start_time = time.perf_counter()
        detections = []
        h, w = frame.shape[:2]

        ALLOW_GT = os.getenv("ALLOW_GT_FALLBACK", "false").lower() == "true"
        if self.use_onnx and self.session is not None:
            from src.vision.quality import check_frame
            q = check_frame(frame)
            if not q["ok"]:
                print(f"[Detector] reject frame: {q['reason']} lap_var={q['laplacian_var']:.1f} bright={q['brightness']:.1f}")
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return self.render_annotations(frame.copy(), [], latency_ms), [], latency_ms
            blob = self.preprocess(frame)
            raw_outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
            detections = self._postprocess_yolov8(raw_outputs, orig_w=w, orig_h=h)
            if not detections and ground_truth_defects and ALLOW_GT:
                detections = ground_truth_defects
        else:
            time.sleep(0.025)
            detections = (ground_truth_defects or []) if ALLOW_GT else []

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        annotated_frame = self.render_annotations(frame.copy(), detections, latency_ms)

        return annotated_frame, detections, latency_ms

    def render_annotations(self, img: np.ndarray, detections: List[Dict[str, Any]], latency_ms: float) -> np.ndarray:
        """Draws bounding boxes, defect labels, and telemetry HUD on the image."""
        fps = 1000.0 / max(latency_ms, 1.0)

        # Draw Telemetry HUD Banner
        cv2.rectangle(img, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (320, 60), (0, 255, 0) if len(detections) == 0 else (0, 0, 255), 2)
        status_text = "STATUS: PASS (NO DEFECTS)" if len(detections) == 0 else f"STATUS: DEFECT DETECTED ({len(detections)})"
        cv2.putText(img, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, f"ONNX CPU: {latency_ms:.1f}ms | {fps:.1f} FPS", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1)

        # Draw Defect Bounding Boxes
        for det in detections:
            cls_name = det.get("class", "defect")
            conf = det.get("confidence", 0.90)
            x1, y1, x2, y2 = det.get("bbox", [100, 100, 200, 200])

            color = self.COLOR_MAP.get(cls_name, (0, 0, 255))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"{cls_name.upper()}: {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (int(x1), max(0, int(y1) - 18)), (int(x1) + tw + 6, int(y1)), color, -1)
            cv2.putText(img, label, (int(x1) + 3, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return img
