"""Ingest real PCB images (Kaggle akhatova/pcb-defects) through the ONNX
detector and store telemetry as InspectionLog rows (Neon production DB).

Usage:
    .venv/bin/python scripts/ingest_pcb_dataset.py --limit 200
    .venv/bin/python scripts/ingest_pcb_dataset.py --limit 50 --dataset-dir /path/to/dataset
"""
import argparse
import datetime
import random
import sys
import uuid
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.database import SessionLocal, engine, init_db  # noqa: E402
from src.backend.models import InspectionLog  # noqa: E402
from src.backend.main import record_inspection  # noqa: E402
from src.backend.schemas import InspectionCreate  # noqa: E402
from src.vision.detector import PCBDefectDetector  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_SLUG = "akhatova/pcb-defects"
TIMESTAMP_SPACING_S = 20


def collect_kaggle_images(root: Path):
    """Scan immediate class subdirs (except images/labels), recursive rglob."""
    out = []
    for cls_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not cls_dir.is_dir() or cls_dir.name in {"images", "labels"}:
            continue
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out.append((p, cls_dir.name))
    return out


def collect_yolo_images(root: Path):
    """Concatenate images/val + images/test + images/train in that order."""
    out = []
    for split in ("val", "test", "train"):
        d = root / "images" / split
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    out.append((p, split))
    return out


def collect_dataset_images(dataset_dir: Path):
    kaggle = collect_kaggle_images(dataset_dir)
    if kaggle:
        return kaggle
    return collect_yolo_images(dataset_dir)


def find_dataset_dir(cli_dir=None):
    if cli_dir:
        p = Path(cli_dir).expanduser()
        if not p.is_dir():
            raise SystemExit(f"--dataset-dir not found: {p}")
        return p
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub missing. Run: uv pip install --python .venv/bin/python kagglehub")
    print(f"[ingest] downloading {DATASET_SLUG} via kagglehub ...")
    return Path(kagglehub.dataset_download(DATASET_SLUG))


def build_record(det, img_rel, line_id, ts, idx, latency_ms):
    """Build an InspectionLog-ready dict; normalizes raw `short` label."""
    raw = str(det.get("class", "defect"))
    cls = PCBDefectDetector.CLASS_ALIASES.get(raw, raw)
    conf = float(det.get("confidence", 0.0))
    bbox = [float(v) for v in det.get("bbox", [])]
    return {
        "inspection_id": f"INSP-KG-{uuid.uuid4().hex[:8].upper()}",
        "line_id": line_id,
        "timestamp": ts - datetime.timedelta(seconds=idx * TIMESTAMP_SPACING_S),
        "image_filename": img_rel,
        "is_defective": True,
        "defect_classes": [cls],
        "confidence_scores": [round(conf, 2)],
        "bounding_boxes": [bbox],
        "inference_time_ms": round(float(latency_ms), 1),
    }


def run_ingest(dataset_dir, limit=200, line_id="SMT-LINE-01", spacing_s=20,
               session_factory=None, detector=None):
    dataset_dir = Path(dataset_dir)
    items = collect_dataset_images(dataset_dir)
    rng = random.Random(42)
    rng.shuffle(items)
    items = items[:limit]
    det = detector or PCBDefectDetector()
    fac = session_factory or SessionLocal
    inserted = skipped = triggered = 0
    tickets = []
    db = fac()
    try:
        for idx, (img_path, _cls) in enumerate(items):
            rel = img_path.relative_to(dataset_dir).as_posix()
            fname = f"pcb-kaggle/{rel}"
            if db.query(InspectionLog).filter_by(image_filename=fname).first():
                skipped += 1
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
            _, dets, lat = det.infer(img)
            ts = datetime.datetime.now(datetime.timezone.utc)
            if dets:
                for d in dets:
                    row = build_record(d, fname, line_id, ts, idx, lat)
                    payload = InspectionCreate(
                        line_id=line_id, is_defective=True,
                        defect_classes=row["defect_classes"],
                        confidence_scores=row["confidence_scores"],
                        bounding_boxes=row["bounding_boxes"],
                        inference_time_ms=row["inference_time_ms"])
                    resp = record_inspection(payload, db=db)
                    log = db.query(InspectionLog).filter_by(inspection_id=resp.inspection_id).first()
                    if log is not None:
                        log.image_filename = fname
                        db.commit()
                    if resp.incident_triggered:
                        triggered += 1
                        if resp.ticket_id:
                            tickets.append(resp.ticket_id)
                    inserted += 1
            else:
                db.add(InspectionLog(
                    inspection_id=f"INSP-KG-{uuid.uuid4().hex[:8].upper()}",
                    line_id=line_id,
                    timestamp=ts - datetime.timedelta(seconds=idx * spacing_s),
                    image_filename=fname, is_defective=False,
                    defect_classes=[], confidence_scores=[], bounding_boxes=[],
                    inference_time_ms=round(float(lat), 1)))
                db.commit()
                inserted += 1
    finally:
        db.close()
    return {"inserted": inserted, "skipped": skipped, "triggered": triggered, "tickets": tickets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--line-id", default="SMT-LINE-01")
    ap.add_argument("--spacing-s", type=int, default=20)
    ap.add_argument("--dataset-dir", default=None)
    args = ap.parse_args()
    init_db()
    ds = find_dataset_dir(args.dataset_dir)
    print(f"[ingest] dataset: {ds} engine: {str(engine.url).split('@')[0]}")
    res = run_ingest(ds, limit=args.limit, line_id=args.line_id, spacing_s=args.spacing_s)
    print(f"[ingest] done: {res}")


if __name__ == "__main__":
    main()

