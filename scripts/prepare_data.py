#!/usr/bin/env python3
"""
Data preparation script for ApexInspect-AI.
Converts various annotation formats to YOLO format.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml


def convert_coco_to_yolo(
    coco_json: Path,
    images_dir: Path,
    output_dir: Path,
    class_mapping: Optional[Dict[int, int]] = None,
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json: Path to COCO annotations JSON
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format
        class_mapping: Map COCO category IDs to YOLO class IDs
        split_ratios: (train, val, test) ratios
        seed: Random seed for splitting
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json) as f:
        coco = json.load(f)
    
    # Build category mapping
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    if class_mapping is None:
        class_mapping = {cat_id: i for i, cat_id in enumerate(sorted(categories.keys()))}
    
    # Group annotations by image
    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)
    
    # Get image list
    images = {img["id"]: img for img in coco["images"]}
    image_ids = list(images.keys())
    
    # Split
    train_ids, temp_ids = train_test_split(
        image_ids, train_size=split_ratios[0], random_state=seed
    )
    val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_ratio, random_state=seed
    )
    
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    
    # Process each split
    for split_name, split_ids in splits.items():
        split_img_dir = output_dir / "images" / split_name
        split_label_dir = output_dir / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id in tqdm(split_ids, desc=f"Processing {split_name}"):
            img_info = images[img_id]
            src_img = images_dir / img_info["file_name"]
            dst_img = split_img_dir / img_info["file_name"]
            
            # Copy image
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                print(f"Warning: Image not found: {src_img}")
                continue
            
            # Convert annotations
            anns = img_anns.get(img_id, [])
            if anns:
                h, w = img_info["height"], img_info["width"]
                yolo_anns = []
                
                for ann in anns:
                    cat_id = ann["category_id"]
                    if cat_id not in class_mapping:
                        continue
                    
                    yolo_class = class_mapping[cat_id]
                    bbox = ann["bbox"]  # [x, y, width, height] in pixels
                    
                    # Convert to YOLO format (normalized center x, y, width, height)
                    x_center = (bbox[0] + bbox[2] / 2) / w
                    y_center = (bbox[1] + bbox[3] / 2) / h
                    width = bbox[2] / w
                    height = bbox[3] / h
                    
                    yolo_anns.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save label file
                label_file = split_label_dir / f"{Path(img_info['file_name']).stem}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_anns))
            else:
                # Empty label file for images without annotations
                label_file = split_label_dir / f"{Path(img_info['file_name']).stem}.txt"
                label_file.touch()
    
    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {v: k for k, v in {cat_id: categories[cat_id] for cat_id in class_mapping}.items()},
    }
    
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\nConversion complete!")
    print(f"Output: {output_dir}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")


def convert_voc_to_yolo(
    voc_dir: Path,
    output_dir: Path,
    class_names: List[str],
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    """Convert Pascal VOC XML annotations to YOLO format."""
    # Implementation similar to COCO but parsing XML
    pass


def create_data_yaml(
    output_dir: Path,
    class_names: List[str],
    train_path: str = "images/train",
    val_path: str = "images/val",
    test_path: str = "images/test",
):
    """Create data.yaml for YOLO training."""
    data = {
        "path": str(output_dir.absolute()),
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "names": {i: name for i, name in enumerate(class_names)},
    }
    
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def verify_yolo_format(data_dir: Path):
    """Verify YOLO format dataset."""
    data_dir = Path(data_dir)
    
    for split in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split
        label_dir = data_dir / "labels" / split
        
        if not img_dir.exists():
            print(f"Missing: {img_dir}")
            continue
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(label_dir.glob("*.txt"))
        
        print(f"\n{split}: {len(images)} images, {len(labels)} labels")
        
        # Check matching
        img_stems = {f.stem for f in images}
        label_stems = {f.stem for f in labels}
        
        missing_labels = img_stems - label_stems
        missing_images = label_stems - img_stems
        
        if missing_labels:
            print(f"  Missing labels for: {missing_labels}")
        if missing_images:
            print(f"  Missing images for: {missing_images}")


def main():
    parser = argparse.ArgumentParser(description="Convert annotations to YOLO format")
    parser.add_argument("--coco-json", type=Path, help="COCO annotations JSON")
    parser.add_argument("--images-dir", type=Path, required=True, help="Images directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--class-mapping", type=Path, help="JSON file for class mapping")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="Verify existing YOLO dataset")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_yolo_format(args.output_dir)
        return
    
    if args.coco_json:
        class_mapping = None
        if args.class_mapping:
            with open(args.class_mapping) as f:
                class_mapping = json.load(f)
        
        convert_coco_to_yolo(
            args.coco_json,
            args.images_dir,
            args.output_dir,
            class_mapping=class_mapping,
            split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed,
        )
    else:
        parser.error("--coco-json is required")


if __name__ == "__main__":
    main()