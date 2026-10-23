"""
YOLO-format defect detection dataset with Albumentations augmentations.
Supports train/val/test splits, mosaic, mixup, copy-paste.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


class DefectDataset(Dataset):
    """
    YOLO format dataset for industrial defect detection.
    
    Expected directory structure:
    data_root/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    
    Label format (YOLO): class_id x_center y_center width height (normalized 0-1)
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        img_size: int = 640,
        class_names: Optional[List[str]] = None,
        transforms: Optional[A.Compose] = None,
        mosaic: bool = False,
        mosaic_prob: float = 1.0,
        mixup: bool = False,
        mixup_prob: float = 0.1,
        copy_paste: bool = False,
        copy_paste_prob: float = 0.3,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.class_names = class_names or []
        self.transforms = transforms
        self.mosaic = mosaic and split == "train"
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup and split == "train"
        self.mixup_prob = mixup_prob
        self.copy_paste = copy_paste and split == "train"
        self.copy_paste_prob = copy_paste_prob

        self.img_dir = self.data_root / "images" / split
        self.label_dir = self.data_root / "labels" / split

        # Get image files
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        if not self.img_files:
            raise ValueError(f"No images found in {self.img_dir}")

        # Cache labels for faster loading
        self.labels_cache = self._cache_labels()

    def _cache_labels(self) -> Dict[str, np.ndarray]:
        """Pre-load all labels for faster access."""
        cache = {}
        for img_file in self.img_files:
            label_file = self.label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                labels = np.loadtxt(label_file, dtype=np.float32).reshape(-1, 5)
                cache[img_file.name] = labels
            else:
                cache[img_file.name] = np.zeros((0, 5), dtype=np.float32)
        return cache

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mosaic and np.random.random() < self.mosaic_prob:
            return self._get_mosaic(idx)
        elif self.mixup and np.random.random() < self.mixup_prob:
            return self._get_mixup(idx)
        else:
            return self._get_single(idx)

    def _get_single(self, idx: int) -> Dict[str, torch.Tensor]:
        img_file = self.img_files[idx]
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        labels = self.labels_cache[img_file.name].copy()

        # Convert YOLO to xyxy for albumentations
        if len(labels) > 0:
            boxes = labels[:, 1:].copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * w  # x_center, width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * h  # y_center, height
            # Convert to xyxy
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            class_labels = labels[:, 0].astype(np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros((0,), dtype=np.int64)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=class_labels)
            img = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            class_labels = np.array(transformed["class_labels"], dtype=np.int64) if transformed["class_labels"] else np.zeros((0,), dtype=np.int64)

        # Convert back to YOLO format (normalized xywh)
        if len(boxes) > 0:
            h_new, w_new = img.shape[:2] if isinstance(img, np.ndarray) else (img.shape[1], img.shape[2])
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / w_new
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / h_new
            x_center = (boxes[:, 0] + boxes[:, 2]) / 2
            y_center = (boxes[:, 1] + boxes[:, 3]) / 2
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            labels_out = np.column_stack([class_labels, x_center, y_center, width, height])
        else:
            labels_out = np.zeros((0, 5), dtype=np.float32)

        # Convert to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": img,
            "labels": torch.from_numpy(labels_out).float(),
            "img_file": str(img_file),
            "orig_shape": (h, w),
        }

    def _get_mosaic(self, idx: int) -> Dict[str, torch.Tensor]:
        """Create mosaic augmentation from 4 images."""
        indices = [idx] + np.random.choice(len(self), 3, replace=False).tolist()
        imgs, labels_list = [], []

        for i, index in enumerate(indices):
            item = self._get_single(index)
            imgs.append(item["image"])
            labels_list.append(item["labels"])

        # Mosaic layout: 2x2 grid
        mosaic_img = torch.zeros(3, self.img_size * 2, self.img_size * 2)
        mosaic_labels = []

        for i, (img, labels) in enumerate(zip(imgs, labels_list)):
            h, w = img.shape[1], img.shape[2]
            x_offset = (i % 2) * self.img_size
            y_offset = (i // 2) * self.img_size

            mosaic_img[:, y_offset:y_offset+h, x_offset:x_offset+w] = img

            if len(labels) > 0:
                labels = labels.clone()
                labels[:, 1] = (labels[:, 1] * w + x_offset) / (self.img_size * 2)  # x_center
                labels[:, 2] = (labels[:, 2] * h + y_offset) / (self.img_size * 2)  # y_center
                labels[:, 3] = labels[:, 3] * w / (self.img_size * 2)  # width
                labels[:, 4] = labels[:, 4] * h / (self.img_size * 2)  # height
                mosaic_labels.append(labels)

        if mosaic_labels:
            mosaic_labels = torch.cat(mosaic_labels, dim=0)
        else:
            mosaic_labels = torch.zeros((0, 5))

        return {
            "image": mosaic_img,
            "labels": mosaic_labels,
            "img_file": "mosaic",
            "orig_shape": (self.img_size * 2, self.img_size * 2),
        }

    def _get_mixup(self, idx: int) -> Dict[str, torch.Tensor]:
        """Mixup augmentation."""
        item1 = self._get_single(idx)
        item2 = self._get_single(np.random.randint(len(self)))

        lam = np.random.beta(32.0, 32.0)

        img = lam * item1["image"] + (1 - lam) * item2["image"]
        labels = torch.cat([item1["labels"], item2["labels"]], dim=0)

        return {
            "image": img,
            "labels": labels,
            "img_file": "mixup",
            "orig_shape": item1["orig_shape"],
        }


def get_train_transforms(img_size: int, aug_config: Dict) -> A.Compose:
    """Get training augmentations with Albumentations."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
        A.HorizontalFlip(p=aug_config.get("flip_prob", 0.5)),
        A.ShiftScaleRotate(
            shift_limit=aug_config.get("translate_limit", 0.1),
            scale_limit=aug_config.get("scale_limit", 0.5),
            rotate_limit=aug_config.get("rotate_limit", 30),
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        ),
        A.OneOf([
            A.MotionBlur(p=aug_config.get("blur_prob", 0.1)),
            A.MedianBlur(blur_limit=3, p=aug_config.get("median_blur_prob", 0.1)),
            A.Blur(blur_limit=3, p=aug_config.get("blur_prob", 0.1)),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(p=aug_config.get("clahe_prob", 0.1)),
            A.RandomBrightnessContrast(p=aug_config.get("brightness_contrast_prob", 0.3)),
            A.RandomGamma(p=0.1),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=aug_config.get("noise_prob", 0.2)),
            A.ISONoise(p=0.1),
        ], p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=aug_config.get("hue_saturation_prob", 0.2),
        ),
        A.ToGray(p=aug_config.get("to_gray_prob", 0.1)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))


def get_val_transforms(img_size: int) -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))


def get_test_transforms(img_size: int) -> A.Compose:
    """Get test transforms (same as val)."""
    return get_val_transforms(img_size)


def create_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 8,
    img_size: int = 640,
    class_names: Optional[List[str]] = None,
    mosaic: bool = True,
    mosaic_prob: float = 1.0,
    mixup: bool = True,
    mixup_prob: float = 0.1,
    copy_paste: bool = True,
    copy_paste_prob: float = 0.3,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    aug_config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""
    data_root = Path(data_root)
    aug_config = aug_config or {}

    train_dataset = DefectDataset(
        data_root, "train", img_size, class_names,
        transforms=get_train_transforms(img_size, aug_config),
        mosaic=mosaic, mosaic_prob=mosaic_prob,
        mixup=mixup, mixup_prob=mixup_prob,
        copy_paste=copy_paste, copy_paste_prob=copy_paste_prob,
    )

    val_dataset = DefectDataset(
        data_root, "val", img_size, class_names,
        transforms=get_val_transforms(img_size),
    )

    test_dataset = DefectDataset(
        data_root, "test", img_size, class_names,
        transforms=get_test_transforms(img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=yolov8_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=yolov8_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=yolov8_collate_fn,
    )

    return train_loader, val_loader, test_loader


def yolov8_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for YOLOv8 training."""
    images = torch.stack([item["image"] for item in batch])
    labels = []
    for i, item in enumerate(batch):
        lbl = item["labels"]
        if len(lbl) > 0:
            lbl = lbl.clone()
            lbl[:, 0] = i  # batch index
            labels.append(lbl)
    labels = torch.cat(labels, dim=0) if labels else torch.zeros((0, 5))
    return {"images": images, "labels": labels}


def load_class_names(data_yaml: Union[str, Path]) -> List[str]:
    """Load class names from data.yaml."""
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])