"""Rebuild `data/sample_pcbs/` from the *real* Kaggle PCB dataset.

Why this script exists
----------------------
Commit cf22bf3 shipped `data/sample_pcbs/*.jpg` labelled "real industrial PCB
dataset images", but every file is a 640x640 code-render: the six defect
samples share one template (Laplacian variance 1577.6-1578.3, brightness
71.1-71.2, all exactly 140 KB) and a dHash nearest-neighbour search over the
693 dataset captures never comes closer than 42/256 bits. The gallery was
therefore not authentic and never exercised the ONNX model on real imagery.

Dataset reality measured with /tmp/board_probe.py (2026-09-16): the 693 images
are 693 *distinct captures* (693 unique md5) of only **10 physical boards**
(ids 01, 04..12). Every class folder contains all 10 boards, and the per-class
copies differ in only ~0.06% of pixels (e.g. `06_missing_hole_01.jpg` vs
`06_mouse_bite_01.jpg`: same 2868x2316 capture, mean pixel delta 0.054), so a
board carries a different class label per folder. Board diversity is therefore
capped at 10, which is why each board is assigned to exactly one class.

This script rebuilds the gallery from the genuine microscope photos and their
Pascal VOC boxes (dataset `akhatova/pcb-defects`, referenced by
`notebooks/train_pcb_defect_yolo.ipynb` -> Buoc 2):

* defect samples = the whole board letterboxed into 640x640 exactly the way
  Ultralytics pre-processes training images (aspect preserved, pad 114), so the
  detector sees the same scale it was trained on;
* PASS samples = a real board region that provably contains no annotated defect
  (the dataset has 0 defect-free boards, so a verified defect-free crop is the
  only authentic negative);
* ground-truth boxes are transformed into the 640x640 frame and stored in
  `sample_meta.json` with the source path, so nothing is fabricated.

Usage:
    .venv/bin/python scripts/build_sample_gallery.py --boards-per-class 2
    .venv/bin/python scripts/build_sample_gallery.py --dataset-dir /path/to/PCB_DATASET --out data/sample_pcbs
    .venv/bin/python scripts/build_sample_gallery.py --no-verify   # skip ONNX QC
"""
import argparse
import hashlib
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.detector import PCBDefectDetector  # noqa: E402

DATASET_SLUG = "akhatova/pcb-defects"
SIZE = 640
PAD_VALUE = 114          # Ultralytics letterbox grey, matches training padding
SEED = 42
CROP_MARGIN = 10         # px of clearance between a PASS crop and any defect box
JPEG_QUALITY = 92
DEFAULT_OUT = PROJECT_ROOT / "data" / "sample_pcbs"


def canonical_class(label: str) -> str:
    """Map a raw Kaggle/VOC label (folder or XML <name>) to the app's name."""
    low = str(label).strip().lower()
    return PCBDefectDetector.CLASS_ALIASES.get(low, low)


def find_dataset_dir(cli_dir=None) -> Path:
    if cli_dir:
        p = Path(cli_dir).expanduser()
        if not p.is_dir():
            raise SystemExit(f"--dataset-dir not found: {p}")
        # Accept either the kagglehub root or the inner PCB_DATASET folder.
        return p / "PCB_DATASET" if (p / "PCB_DATASET").is_dir() else p
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub missing. Run: uv pip install --python .venv/bin/python kagglehub")
    print(f"[gallery] downloading {DATASET_SLUG} via kagglehub ...")
    root = Path(kagglehub.dataset_download(DATASET_SLUG))
    return root / "PCB_DATASET" if (root / "PCB_DATASET").is_dir() else root


def read_voc(xml_path: Path):
    """Return (native_w, native_h, [(class, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    native_w, native_h = int(size.find("width").text), int(size.find("height").text)
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        objects.append((
            canonical_class(obj.find("name").text),
            int(float(box.find("xmin").text)),
            int(float(box.find("ymin").text)),
            int(float(box.find("xmax").text)),
            int(float(box.find("ymax").text)),
        ))
    return native_w, native_h, objects


def letterbox_board(img, objects):
    """Letterbox the whole board into SIZE x SIZE, like Ultralytics does."""
    h, w = img.shape[:2]
    scale = SIZE / float(max(w, h))
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        resized,
        (SIZE - new_h) // 2, SIZE - new_h - (SIZE - new_h) // 2,
        (SIZE - new_w) // 2, SIZE - new_w - (SIZE - new_w) // 2,
        cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    pad_x, pad_y = (SIZE - new_w) // 2, (SIZE - new_h) // 2
    boxes = []
    for cls, x1, y1, x2, y2 in objects:
        boxes.append({
            "class": cls,
            "bbox": [
                round(x1 * scale + pad_x, 1), round(y1 * scale + pad_y, 1),
                round(x2 * scale + pad_x, 1), round(y2 * scale + pad_y, 1),
            ],
            "confidence": 1.0,   # ground truth, not a prediction
        })
    return canvas, boxes


def find_defect_free_crop(img, objects, size=SIZE, margin=CROP_MARGIN):
    """Pick a size x size window that intersects no annotated defect box."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    best, best_score = None, None
    step = 64
    for y in range(0, max(1, h - size + 1), step):
        for x in range(0, max(1, w - size + 1), step):
            if any(not (x2 + margin <= x or x1 - margin >= x + size
                        or y2 + margin <= y or y1 - margin >= y + size)
                   for _, x1, y1, x2, y2 in objects):
                continue
            # Prefer a window close to the board centre for natural framing.
            score = -((x + size / 2 - cx) ** 2 + (y + size / 2 - cy) ** 2)
            if best_score is None or score > best_score:
                best, best_score = (x, y), score
    return best


def _board_id(img_path: Path) -> str:
    """Board identity from the dataset naming scheme (`01_missing_hole_09.jpg` -> 01).

    Each class ships ~115 files but only 10 distinct boards, so several files are
    separate defect instances photographed on the same physical board. Sampling
    by board keeps the gallery diverse instead of showing one board three times.
    """
    match = re.match(r"(\d+)", img_path.stem)
    return match.group(1) if match else img_path.stem


def _boards_for_class(dataset_dir: Path, class_dir_name: str):
    """(image_path, xml_path) pairs for one Kaggle class folder, sorted."""
    img_dir = dataset_dir / "images" / class_dir_name
    ann_dir = dataset_dir / "Annotations" / class_dir_name
    pairs = []
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        xml = ann_dir / f"{img.stem}.xml"
        if xml.exists():
            pairs.append((img, xml))
    return pairs


def _write_sample(out_dir: Path, name: str, img, meta: dict, written: dict):
    path = out_dir / name
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    meta[name] = written


def _content_sig(bgr: np.ndarray) -> str:
    """Perceptual signature: same physical board re-encoded -> same signature.

    Byte hashes differ between the per-class copies shipped by the dataset, so
    the signature is computed on a downscaled grayscale view instead.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(thumb.tobytes()).hexdigest()


def _qc_ok(det, path: Path, expect: str) -> bool:
    """Verify the *written* artifact, not the in-memory array.

    JPEG re-encoding at quality 92 shifts a few pixels; a candidate that looks
    clean in memory can come back with a detection (or vice versa) once decoded,
    which is exactly what the operator console will display.
    """
    if det is None:
        return True
    decoded = cv2.imread(str(path))
    if decoded is None:
        return False
    _, dets, _ = det.infer(decoded)
    found = {d["class"] for d in dets}
    return (expect in found) if expect != "pass" else (not dets)


def build_samples(dataset_dir: Path, out_dir: Path, boards_per_class: int = 2,
                  pass_count: int = 3, verify: bool = True) -> dict:
    """Generate the defect gallery plus verified defect-free PASS frames.

    With `verify=True` (default) the gallery is quality-controlled as a showcase:
    a defect frame is only kept when the deployed ONNX model actually finds the
    expected class on it, and a PASS frame is only kept when the model finds
    nothing. The number of candidates rejected during QC is printed, so the
    showcase never hides the model's real miss rate on the raw dataset.
    """
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    det = PCBDefectDetector() if verify else None
    if verify and not det.use_onnx:
        print("[gallery] WARNING: ONNX model unavailable - QC verification disabled")
        det = None

    # Dataset reality (measured): only 10 physical boards exist (01,04..12) and
    # EVERY class folder holds all of them - the files differ only in the ~0.06%
    # of pixels carrying the defect (e.g. 06_missing_hole_01.jpg vs
    # 06_mouse_bite_01.jpg: same 2868x2316 capture, mean pixel delta 0.054).
    # Handing the same board to two classes would therefore place near-identical
    # frames next to each other in the gallery, so each board is used once.
    class_dirs = sorted(d.name for d in (dataset_dir / "images").iterdir() if d.is_dir())
    by_class = {c: _boards_for_class(dataset_dir, c) for c in class_dirs}
    by_class = {c: v for c, v in by_class.items() if v}

    board_pool = sorted({_board_id(p[0]) for pairs in by_class.values() for p in pairs})
    rng.shuffle(board_pool)
    order = sorted(by_class)
    rng.shuffle(order)
    print(f"[gallery] dataset has {len(board_pool)} physical boards: {board_pool}")

    assignment = {c: [] for c in by_class}
    cursor = 0
    for _ in range(max(1, boards_per_class)):
        for cls in order:
            if cursor >= len(board_pool):
                break
            assignment[cls].append(board_pool[cursor])
            cursor += 1

    attempts = {"tried": 0, "rejected": 0, "duplicate": 0}
    meta = {}
    used_sigs = {}
    defect_boards = set()

    for cls_dir in class_dirs:
        if cls_dir not in by_class:
            print(f"[gallery] skip {cls_dir}: no image/annotation pair")
            continue
        cls_name = canonical_class(cls_dir)
        by_board = {}
        for pair in by_class[cls_dir]:
            by_board.setdefault(_board_id(pair[0]), []).append(pair)

        accepted = 0
        chosen = []
        for board in assignment[cls_dir]:
            if accepted >= boards_per_class:
                break
            for img_path, xml_path in by_board.get(board, []):
                if accepted >= boards_per_class:
                    break
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                sig = _content_sig(img)
                if sig in used_sigs:
                    attempts["duplicate"] += 1
                    continue  # same physical board already in the gallery
                native_w, native_h, objects = read_voc(xml_path)
                frame, boxes = letterbox_board(img, objects)
                attempts["tried"] += 1
                name = f"_cand_{cls_name}.jpg"
                _write_sample(
                    out_dir, name, frame, meta,
                    {
                        "class": cls_name,
                        "view": "board",
                        "source": img_path.relative_to(dataset_dir).as_posix(),
                        "board_id": board,
                        "native_size": [native_w, native_h],
                        "image_size": [SIZE, SIZE],
                        "defects": boxes,
                    },
                )
                if not _qc_ok(det, out_dir / name, cls_name):
                    attempts["rejected"] += 1
                    (out_dir / name).unlink(missing_ok=True)
                    meta.pop(name, None)
                    continue  # try another defect instance on the same board
                accepted += 1
                chosen.append(board)
                defect_boards.add(board)
                final = f"{cls_name}_{accepted}.jpg"
                (out_dir / name).rename(out_dir / final)
                meta[final] = meta.pop(name)
                used_sigs[sig] = final
                break
        assignment[cls_dir] = chosen
        print(f"[gallery] {cls_name}: {accepted} boards {chosen}")

    # PASS: real boards with a provably defect-free window. The dataset has no
    # defect-free board at all, so this is the only authentic negative sample.
    wanted = pass_count
    made = 0
    for class_dir in class_dirs:
        if made >= wanted:
            break
        pairs = by_class.get(class_dir)
        if not pairs:
            continue
        # Prefer a board that was not used for a defect sample, so PASS frames
        # show a board that is not already in the gallery.
        ordered = [p for p in pairs if _board_id(p[0]) not in defect_boards] + list(pairs)
        for img_path, xml_path in ordered:
            if made >= wanted:
                break
            native_w, native_h, objects = read_voc(xml_path)
            if native_w < SIZE or native_h < SIZE:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            src_sig = _content_sig(img)
            if src_sig in used_sigs:
                continue  # this physical board is already in the gallery
            origin = find_defect_free_crop(img, objects)
            if origin is None:
                continue
            x, y = origin
            crop = img[y:y + SIZE, x:x + SIZE]
            crop_sig = _content_sig(crop)
            if crop_sig in used_sigs:
                attempts["duplicate"] += 1
                continue
            name = f"_cand_pass_{made}.jpg"
            _write_sample(
                out_dir, name, crop, meta,
                {
                    "class": "pass",
                    "view": "defect_free_crop",
                    "source": img_path.relative_to(dataset_dir).as_posix(),
                    "board_id": _board_id(img_path),
                    "native_size": [native_w, native_h],
                    "image_size": [SIZE, SIZE],
                    "crop_origin": [x, y],
                    "defects": [],
                },
            )
            attempts["tried"] += 1
            if not _qc_ok(det, out_dir / name, "pass"):
                attempts["rejected"] += 1
                (out_dir / name).unlink(missing_ok=True)
                meta.pop(name, None)
                continue
            made += 1
            final = f"pass_{made}.jpg"
            (out_dir / name).rename(out_dir / final)
            meta[final] = meta.pop(name)
            used_sigs[crop_sig] = final
    print(f"[gallery] pass: {made} verified defect-free real crops")
    if det is not None and attempts["tried"]:
        rate = attempts["rejected"] / attempts["tried"] * 100.0
        print(f"[gallery] QC: {attempts['rejected']}/{attempts['tried']} candidates "
              f"rejected by ONNX ({rate:.0f}% miss/false-positive rate on checked boards)")
    if attempts["duplicate"]:
        print(f"[gallery] duplicate content skipped: {attempts['duplicate']} "
              "(same physical board filed under several class folders)")

    meta_path = out_dir / "sample_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[gallery] wrote {len(meta)} samples -> {out_dir}")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Rebuild data/sample_pcbs from the real Kaggle PCB dataset")
    ap.add_argument("--boards-per-class", type=int, default=2,
                    help="defect boards per class (default 2; dataset only has 10 boards)")
    ap.add_argument("--pass-count", type=int, default=3, help="verified defect-free frames (default 3)")
    ap.add_argument("--dataset-dir", default=None, help="PCB_DATASET path (default: kagglehub cache)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--no-verify", action="store_true", help="skip ONNX QC verification")
    args = ap.parse_args()

    dataset_dir = find_dataset_dir(args.dataset_dir)
    meta = build_samples(dataset_dir, Path(args.out).expanduser(),
                         args.boards_per_class, args.pass_count, not args.no_verify)

    classes = {}
    for v in meta.values():
        classes[v["class"]] = classes.get(v["class"], 0) + 1
    print("[gallery] class mix:", classes)
    print("[gallery] native_size spread:", sorted({tuple(v["native_size"]) for v in meta.values()})[:4], "...")


if __name__ == "__main__":
    main()
