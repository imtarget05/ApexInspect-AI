# Model Card — `yolov8n_pcb_defect.onnx` (ApexInspect AI)

Nguon train: Google Colab GPU T4, notebook `notebooks/train_pcb_defect_yolo.ipynb`
(export tu `notebooks/best.onnx`, giu nguyen trong so).

- Base: `yolov8n.pt` (Ultralytics 8.4.153), fine-tune PCB defect 50 epochs,
  imgsz 640, batch 16, patience 10, optimizer AdamW lr0 0.001.
- Dataset: `akhatova/pcb-defects` (Kaggle, qua `kagglehub`), convert Pascal VOC
  XML -> YOLO TXT, chia `images/train` + `images/val` (`models/training_data.yaml`).
- 6 classes (thu tu output ONNX): `missing_hole` (0), `mouse_bite` (1),
  `open_circuit` (2), `short` (3), `spur` (4), `spurious_copper` (5).
  Luu y: label goc `short` duoc Detector chuan hoa thanh `short_circuit`
  (ten canonical dung trong simulator / backend / SOP RAG).
- Export: `YOLO(best.pt).export(format="onnx", imgsz=640)`.
- IO da verify bang onnxruntime CPU: input `images [1, 3, 640, 640] float32`,
  output `output0 [1, 10, 8400] float32` = 4 box (cx, cy, w, h) + 6 scores,
  khop `PCBDefectDetector._postprocess_yolov8`.
- File canonical duoc load: `models/yolov8n_pcb_defect.onnx` (~12 MB),
  theo thu tu uu tien `resolve_model_path()`: arg -> `$MODEL_PATH`
  (`.env: MODEL_PATH=models/yolov8n_pcb_defect.onnx`) -> `models/...` ->
  fallback `notebooks/best.onnx`.
- Nguong mac dinh: `CONFIDENCE_THRESHOLD=0.50` (spec FR-1), NMS IoU 0.45,
  latency muc tieu <= 35 ms (~28 FPS, NFR >= 25 FPS).
- Evidence huan luyen: `models/confusion_matrix.png`, `models/results.png`.
- Trien khai: file ONNX duoc git track (exception trong `.gitignore`);
  `best.pt` (6 MB) van bi ignore, giu lai trong `notebooks/` de train tiep.
