#!/usr/bin/env python3
"""Shared utilities for manuscript robustness experiments.

These helpers intentionally mirror the existing root-level scripts rather than
replacing them. They provide reusable pieces for repeated-split training,
leakage audits, and calibration/OOD evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFile, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ["train", "val", "test"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    family: str
    image_size: int


MODEL_CONFIGS = {
    "resnet50": ModelConfig("resnet50", "ResNet50", "torchvision_resnet50", 224),
    "efficientnet_b4": ModelConfig("efficientnet_b4", "EfficientNetB4", "timm_efficientnet_b4", 224),
    "convnext_tiny": ModelConfig("convnext_tiny", "ConvNeXtTiny", "torchvision_convnext_tiny", 224),
    "vit_b16": ModelConfig("vit_b16", "ViTB16", "timm_vit_b16_384_in21k", 384),
}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    class_name: str
    group_id: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def file_md5(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def normalize_stem(stem: str) -> str:
    stem = re.sub(r"\s+", " ", stem.strip())
    stem = re.sub(r"\s*\.?\s*\(\d+\)\s*$", "", stem).strip()
    stem = stem.rstrip(". ").strip()
    return stem


def extract_specimen_group_id(image_path: Path, class_name: str) -> str:
    """Parse a specimen/acquisition group from the filename.

    The current dataset uses names such as
    "3358. Eucalyptus camandulensis.1.(15).png"; removing the final image
    index gives the specimen/acquisition group. The class prefix is retained
    to avoid accidental collisions across species.
    """
    group = normalize_stem(image_path.stem)
    return f"{class_name}::{group or image_path.stem}"


def class_from_relative(path: Path, root: Path) -> str | None:
    rel = path.relative_to(root)
    if len(rel.parts) < 2:
        return None
    return rel.parts[0]


def collect_images_from_root(data_root: Path) -> list[ImageRecord]:
    """Collect records from either a raw class-folder root or split-folder root."""
    data_root = data_root.resolve()
    records: list[ImageRecord] = []

    has_split_dirs = all((data_root / split).is_dir() for split in SPLITS)
    if has_split_dirs:
        for split in SPLITS:
            split_root = data_root / split
            for path in sorted(p for p in split_root.rglob("*") if is_image(p)):
                class_name = class_from_relative(path, split_root)
                if class_name is None:
                    continue
                records.append(
                    ImageRecord(
                        image_path=path,
                        class_name=class_name,
                        group_id=extract_specimen_group_id(path, class_name),
                    )
                )
    else:
        for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
            for path in sorted(p for p in class_dir.rglob("*") if is_image(p)):
                class_name = class_dir.name
                records.append(
                    ImageRecord(
                        image_path=path,
                        class_name=class_name,
                        group_id=extract_specimen_group_id(path, class_name),
                    )
                )

    if not records:
        raise FileNotFoundError(f"No image files found under {data_root}")
    return records


def records_to_dataframe(records: Iterable[ImageRecord]) -> pd.DataFrame:
    rows = [
        {
            "image_path": str(record.image_path),
            "class_name": record.class_name,
            "group_id": record.group_id,
        }
        for record in records
    ]
    df = pd.DataFrame(rows)
    classes = sorted(df["class_name"].unique())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    df["label"] = df["class_name"].map(class_to_idx).astype(int)
    return df[["image_path", "label", "class_name", "group_id"]]


def load_split_file(split_file: Path) -> pd.DataFrame:
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    df = pd.read_csv(split_file)
    required = {"image_path", "label", "class_name", "group_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_file} is missing columns: {sorted(missing)}")
    return df


def infer_classes_from_split(split_df: pd.DataFrame) -> list[str]:
    label_class = split_df[["label", "class_name"]].drop_duplicates().sort_values("label")
    return label_class["class_name"].tolist()


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ManifestImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(row["image_path"])
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError) as exc:
            print(f"[WARN] Could not read image, using blank fallback: {path} ({exc})", flush=True)
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        return self.transform(image), int(row["label"]), str(path)


class UnlabeledImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: transforms.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError) as exc:
            print(f"[WARN] Could not read OOD image, using blank fallback: {path} ({exc})", flush=True)
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        return self.transform(image), str(path)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_model(model_name: str, num_classes: int, pretrained: bool = True, image_size: int | None = None) -> nn.Module:
    if model_name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b4":
        import timm

        return timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=num_classes)

    if model_name == "vit_b16":
        import timm

        names = [
            "vit_base_patch16_384.augreg_in21k",
            "vit_base_patch16_384.orig_in21k",
            "vit_base_patch16_384.augreg_in21k_ft_in1k",
            "vit_base_patch16_384.orig_in21k_ft_in1k",
            "vit_base_patch16_384",
        ]
        last_error: Exception | None = None
        for timm_name in names:
            try:
                kwargs = {"img_size": image_size} if image_size is not None else {}
                return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not create ViT-B/16 model: {last_error}")

    raise ValueError(f"Unsupported model: {model_name}. Choices: {sorted(MODEL_CONFIGS)}")


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_state(model: nn.Module, checkpoint_path: Path, device: torch.device, strict: bool = True) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {"model_state": checkpoint}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    rows = []
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.array([(y_true == idx).sum() for idx in labels])
    for idx, class_name in enumerate(class_names):
        rows.append(
            {
                "label": idx,
                "class_name": class_name,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
                "support": int(support[idx]),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_manifest(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_paths: list[str] = []
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(probs.argmax(dim=1).detach().cpu().numpy().tolist())
        all_probs.append(probs.detach().cpu().numpy())
        all_paths.extend(paths)
    return np.array(all_labels), np.array(all_preds), np.concatenate(all_probs, axis=0), all_paths


def save_confusion_outputs(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_csv: Path, out_png: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_csv)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def dataframe_to_latex_booktabs(df: pd.DataFrame, caption: str, label: str, float_format: str = "%.4f") -> str:
    return df.to_latex(index=False, escape=False, caption=caption, label=label, float_format=lambda x: float_format % x)


def mean_std_summary(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        values = df[metric].dropna().astype(float)
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def save_contact_sheet(
    query_path: Path,
    neighbor_path: Path,
    title: str,
    output_path: Path,
    size: int = 280,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for path in [query_path, neighbor_path]:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), "white")
            offset = ((size - image.width) // 2, (size - image.height) // 2)
            canvas.paste(image, offset)
            images.append(canvas)
    title_h = 70
    sheet = Image.new("RGB", (size * 2, size + title_h), "white")
    sheet.paste(images[0], (0, title_h))
    sheet.paste(images[1], (size, title_h))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title[:180], fill="black")
    draw.text((8, title_h - 20), "Query", fill="black")
    draw.text((size + 8, title_h - 20), "Nearest / candidate", fill="black")
    sheet.save(output_path)
