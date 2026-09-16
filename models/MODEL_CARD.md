# Model Card — `yolov8n_pcb_defect.onnx` (ApexInspect AI)

Nguon train: Google Colab GPU T4, notebook `notebooks/train_pcb_defect_yolo.ipynb`
(export tu `models/yolov8n_pcb_defect.pt`, giu nguyen trong so).

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
  (`.env: MODEL_PATH=models/yolov8n_pcb_defect.onnx`) -> `models/...`.
- Nguong mac dinh: `CONFIDENCE_THRESHOLD=0.50` (spec FR-1), NMS IoU 0.45,
  latency muc tieu <= 35 ms (~28 FPS, NFR >= 25 FPS).
- Evidence huan luyen: `models/confusion_matrix.png`, `models/results.png`.
- Trien khai: file ONNX duoc git track (exception trong `.gitignore`); trong so
  goc `models/yolov8n_pcb_defect.pt` (6 MB) van bi ignore, giu canh ONNX de train tiep.
- Validation: Kaggle val + `images/production_val` (tach khoi train).
- Fallback GT chi khi `ALLOW_GT_FALLBACK=true` (mac dinh false - xem `.env.example`).

## Dataset reality (do luong 2026-09-16)

So lieu do truc tiep tu `akhatova/pcb-defects`; day la ly do gallery mau buoc phai
qua QC bang chinh ONNX truoc khi hien len console:

- 693 anh + 693 XML; board goc 2240x2016 .. 3056x2464 (10 kich thuoc vat ly).
- **0 board khong co loi**: moi XML co 1..6 object (trung binh 4.26), nen mau PASS
  "that" chi co the la mot vung crop da kiem chung khong co annotation.
- **Chi ~10 board vat ly** (01, 04..12) va moi class folder chua lai dung 10 board
  do; file khac nhau chu yeu o vung loi rat nho (`06_missing_hole_01.jpg` vs
  `06_mouse_bite_01.jpg`: cung capture 2868x2316, mean pixel delta ~0.054, tuc
  ~0.06% pixel). Gallery vi vay phai dedupe theo content signature.
- Defect box goc: width 24..283 px (median 64), height 23..215 px (median 63);
  sau letterbox 640 con ~14 px -> chi ro khi co bounding box.
- QC tren 15 ung vien board: 2 bi loai (13%) do model bo sot/bao nham. Con so nay
  la recall thuc te, khong duoc che di.
