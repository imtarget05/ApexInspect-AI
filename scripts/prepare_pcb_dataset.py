"""Build a YOLOv8 dataset from the Kaggle PCB defects collection (VOC -> YOLO).

This is the reproducible, repo-tracked version of what used to live only in
`notebooks/train_pcb_defect_yolo.ipynb` (cell 4). The notebook version had two
problems this script fixes:

1. It wrote only `images/train` + `images/val` while the tracked SoT
   `models/training_data.yaml` declares `test: images/production_val` - a
   split no code could produce.
2. It split by *image* (`random.shuffle(all_samples)` 80/20). Every physical
   board is filed under all six class folders (693 captures of ~10 boards),
   so the same board landed in train AND val and inflated validation scores.

This script splits by *physical board* (parsed from the `NN_` filename
prefix), so no board leaks across splits.

Layout in:  <src>/images/<Class>/*.jpg + <src>/Annotations/<Class>/*.xml
Layout out: <out>/images/{train,val,production_val} + labels/... + data.yaml
"""
import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# Raw Kaggle/VOC labels seen in the wild map to the canonical training ids.
CLASS_TO_ID["short_circuit"] = CLASS_TO_ID["short"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "production_val")


def _board_id(stem: str) -> str:
    """Physical board identity from the dataset naming scheme.

    `01_missing_hole_09` -> `01`. Falls back to the full stem when the name
    does not start with digits (synthetic test boards, custom captures).
    """
    match = re.match(r"(\d+)", stem)
    return match.group(1) if match else stem


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def voc_to_yolo_lines(xml_path, img_w, img_h):
    """Convert one Pascal VOC file to YOLO label lines.

    Returns a list of `"cls_id x_center y_center w h"` strings with
    coordinates normalised to [0, 1]. Unknown classes are dropped; boxes
    sticking out of the frame are clamped (same behaviour as the notebook's
    `convert_xml_to_yolo`, minus the silent `try/except: return False`).
    """
    root = ET.parse(str(xml_path)).getroot()
    lines = []
    for obj in root.iter("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        cls_name = name_el.text.strip().lower()
        if cls_name not in CLASS_TO_ID:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue
        x = _clamp01((xmin + xmax) / 2.0 / img_w)
        y = _clamp01((ymin + ymax) / 2.0 / img_h)
        w = _clamp01((xmax - xmin) / img_w)
        h = _clamp01((ymax - ymin) / img_h)
        lines.append(f"{CLASS_TO_ID[cls_name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines

def _collect_pairs(src: Path):
    """(image_path, xml_path) pairs with a matching VOC annotation."""
    img_root = src / "images"
    ann_root = src / "Annotations"
    if not img_root.is_dir():  # accept the kagglehub root one level up
        nested = src / "PCB_DATASET"
        if (nested / "images").is_dir():
            img_root, ann_root = nested / "images", nested / "Annotations"
    pairs = []
    if not img_root.is_dir():
        return pairs
    for class_dir in sorted(img_root.iterdir(), key=lambda p: p.name):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            xml_path = ann_root / class_dir.name / f"{img_path.stem}.xml"
            if xml_path.exists():
                pairs.append((img_path, xml_path))
    return pairs


def _xml_size(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    size = root.find("size")
    return int(size.find("width").text), int(size.find("height").text)


def build_yolo_dataset(src_dir, out_dir, val_ratio=0.2, prod_ratio=0.2, seed=42):
    """Materialise the board-grouped YOLO dataset; returns a summary dict."""
    src, out = Path(src_dir), Path(out_dir)
    pairs = _collect_pairs(src)
    by_board = {}
    for img_path, xml_path in pairs:
        by_board.setdefault(_board_id(img_path.stem), []).append((img_path, xml_path))

    boards_by_split = split_boards(sorted(by_board), val_ratio=val_ratio,
                                   prod_ratio=prod_ratio, seed=seed)

    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in SPLITS:
        count = 0
        for board in boards_by_split[split]:
            for img_path, xml_path in by_board.get(board, []):
                w, h = _xml_size(xml_path)
                lines = voc_to_yolo_lines(xml_path, w, h)
                shutil.copy2(img_path, out / "images" / split / img_path.name)
                (out / "labels" / split / f"{img_path.stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8")
                count += 1
        counts[split] = count

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/production_val\n"
        "\n"
        "names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )
    return {"boards_by_split": boards_by_split, "counts": counts,
            "out_dir": str(out)}


def split_boards(boards, val_ratio=0.2, prod_ratio=0.2, seed=42):
    """Deterministically split board ids into train/val/production_val.

    Group split (not image split): every board id lands in exactly one split,
    so near-duplicate captures of the same physical board can never leak
    between train and validation. Each split gets at least one board whenever
    there are 3+ boards to distribute.
    """
    shuffled = list(boards)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "production_val": []}
    if n < 3:
        # Not enough boards for three non-empty splits; everything trains.
        return {"train": list(shuffled), "val": [], "production_val": []}
    n_val = min(max(1, int(n * val_ratio)), n - 2)
    n_prod = min(max(1, int(n * prod_ratio)), n - n_val - 1)
    return {
        "train": shuffled[: n - n_val - n_prod],
        "val": shuffled[n - n_val - n_prod: n - n_prod],
        "production_val": shuffled[n - n_prod:],
    }


def main():
    ap = argparse.ArgumentParser(description="Build board-grouped YOLO dataset from Kaggle PCB defects")
    ap.add_argument("--src", required=True, help="dataset root (has images/ + Annotations/)")
    ap.add_argument("--out", default="dataset_pcb", help="output YOLO dataset dir")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--prod-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    summary = build_yolo_dataset(args.src, args.out, args.val_ratio, args.prod_ratio, args.seed)
    print(f"[prepare] boards: { {k: len(v) for k, v in summary['boards_by_split'].items()} }")
    print(f"[prepare] images: {summary['counts']} -> {summary['out_dir']}")


if __name__ == "__main__":
    main()
