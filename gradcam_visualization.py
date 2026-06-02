#!/usr/bin/env python3
"""Create publication-quality Grad-CAM figures for wood species classifiers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from einops import rearrange
except ModuleNotFoundError:
    rearrange = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CKPT_DIR = BASE_DIR / "checkpoints"
GRADCAM_DIR = RESULTS_DIR / "gradcam_individual"
for d in [FIGURES_DIR, GRADCAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---
BEST_MODEL = "convnext_tiny"
SEED = 42
DEFAULT_BATCH_SIZE = 256
USE_AMP = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SHORT_NAMES = {
    "Eucalyptus_grandis": "E. grandis",
    "Eucalyptus_microcorys": "E. microcorys",
    "Eucalyptus_saligna": "E. saligna",
    "Eucalyptus_deglupta": "E. deglupta",
    "Eucalyptus_daglupta": "E. daglupta",
    "Eucalyptus_diversicolor": "E. diversicolor",
    "Eucalyptus_cladocalyx": "E. cladocalyx",
    "Eucalyptus_camaldulensis": "E. camaldulensis",
    "Eucalyptus_camandulensis": "E. camandulensis",
    "Syzygium_hemisphericum": "Syzygium",
}


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    family: str
    image_size: int


@dataclass
class PredictionRecord:
    path: Path
    true_idx: int
    pred_idx: int
    confidence: float
    true_confidence: float


MODEL_CONFIGS = {
    "resnet50": ModelConfig("resnet50", "ResNet50", "torchvision_resnet50", 224),
    "efficientnet_b4": ModelConfig("efficientnet_b4", "EfficientNetB4", "timm_efficientnet_b4", 224),
    "convnext_tiny": ModelConfig("convnext_tiny", "ConvNeXtTiny", "torchvision_convnext_tiny", 224),
    "vit_b16": ModelConfig("vit_b16", "ViTB16", "timm_vit_b16_384_in21k", 384),
}
MODEL_ORDER = ["resnet50", "efficientnet_b4", "convnext_tiny", "vit_b16"]


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that returns the sample index so paths stay aligned."""

    def __init__(self, root: Path, classes: list[str] | None = None, **kwargs):
        self.fixed_classes = classes
        super().__init__(root=str(root), **kwargs)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        if self.fixed_classes is None:
            return super().find_classes(directory)
        return self.fixed_classes, {class_name: idx for idx, class_name in enumerate(self.fixed_classes)}

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, label, index


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Grad-CAM generation on the RTX 3090 server.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[Setup] GPU: {gpu_name} | VRAM: {total_vram_gb:.2f} GB", flush=True)
    return device


def load_label_map() -> dict[int, str]:
    label_map_path = RESULTS_DIR / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_map_path}")
    with label_map_path.open("r", encoding="utf-8") as file:
        raw_map = json.load(file)
    return {int(idx): class_name for idx, class_name in raw_map.items()}


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataset_and_loader(image_size: int, expected_classes: list[str], batch_size: int) -> tuple[ImageFolderWithPaths, DataLoader]:
    test_dir = DATA_DIR / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test split: {test_dir}")

    dataset = ImageFolderWithPaths(test_dir, classes=expected_classes, transform=eval_transform(image_size))
    if dataset.classes != expected_classes:
        raise ValueError(
            "ImageFolder class order does not match results/label_map.json. "
            f"ImageFolder={dataset.classes}, label_map={expected_classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataset, loader


def build_model(config: ModelConfig, num_classes: int) -> nn.Module:
    if config.family == "torchvision_resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if config.family == "torchvision_convnext_tiny":
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if config.family == "timm_efficientnet_b4":
        import timm

        return timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)

    if config.family == "timm_vit_b16_384_in21k":
        import timm

        for model_name in (
            "vit_base_patch16_384.augreg_in21k",
            "vit_base_patch16_384.orig_in21k",
            "vit_base_patch16_384.augreg_in21k_ft_in1k",
            "vit_base_patch16_384.orig_in21k_ft_in1k",
            "vit_base_patch16_384",
        ):
            try:
                return timm.create_model(model_name, pretrained=False, num_classes=num_classes, img_size=384)
            except Exception:
                continue
        raise RuntimeError("Could not create ViT-B/16 model with installed timm.")

    raise ValueError(f"Unknown model family: {config.family}")


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def resolve_checkpoint(model_name: str, config: ModelConfig) -> Path:
    candidates = [
        CKPT_DIR / f"{model_name}_best.pth",
        CKPT_DIR / f"{config.display_name}_best.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate != candidates[0]:
                print(f"[Checkpoint] Using fallback checkpoint: {candidate}", flush=True)
            return candidate
    raise FileNotFoundError(f"No best checkpoint found. Checked: {candidates}")


def load_model(model_name: str, config: ModelConfig, num_classes: int, device: torch.device) -> nn.Module:
    checkpoint_path = resolve_checkpoint(model_name, config)
    print(f"[Checkpoint] Loading {checkpoint_path}", flush=True)
    model = build_model(config, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model


def get_target_layer(model: nn.Module, model_name: str):
    if model_name == "convnext_tiny":
        return model.features[7][2]
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "efficientnet_b4":
        if hasattr(model, "features"):
            return model.features[-1][0]
        return model.conv_head
    elif model_name == "vit_b16":
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers[-1].ln_1
        if hasattr(model, "blocks"):
            return model.blocks[-1].norm1
    raise ValueError(f"Could not auto-select target layer for {model_name}")


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    patch_tokens = tensor[:, 1:, :]
    n_tokens = patch_tokens.shape[1]
    h = int(math.sqrt(n_tokens))
    if h * h != n_tokens:
        h = 14
    if rearrange is not None:
        return rearrange(patch_tokens, "b (h w) c -> b c h w", h=h)
    b, _, c = patch_tokens.shape
    return patch_tokens.reshape(b, h, h, c).permute(0, 3, 1, 2)


@torch.no_grad()
def predict_all(
    model: nn.Module,
    dataset: ImageFolderWithPaths,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict] = []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Extracting predictions", leave=True)

    print(f"[Predictions] Running batched test-set inference with batch_size={loader.batch_size}", flush=True)
    for images, labels, indices in iterator:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=USE_AMP):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        for label, index, pred, conf, prob_row in zip(labels, indices, preds.cpu(), confs.cpu(), probs.cpu()):
            rows.append(
                {
                    "path": dataset.samples[int(index)][0],
                    "true_idx": int(label),
                    "pred_idx": int(pred),
                    "confidence": float(conf),
                    "true_confidence": float(prob_row[int(label)]),
                    "correct": int(label) == int(pred),
                }
            )

    pred_df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "gradcam_predictions.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"[Predictions] Saved prediction table: {output_path}", flush=True)
    return pred_df


def load_display_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return np.asarray(img).astype(np.float32) / 255.0


def tensor_for_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    tensor = eval_transform(image_size)(img).unsqueeze(0).to(device)
    return tensor


def make_cam(
    cam: GradCAM,
    input_tensor: torch.Tensor,
    target_category: int,
    rgb_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])[0]
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return grayscale_cam, overlay


def make_cams_batch(
    cam: GradCAM,
    samples: list[pd.Series],
    config: ModelConfig,
    device: torch.device,
) -> list[dict]:
    """Compute Grad-CAM for a selected set of samples in one batch."""
    if not samples:
        return []

    rgbs: list[np.ndarray] = []
    tensors: list[torch.Tensor] = []
    targets: list[ClassifierOutputTarget] = []

    for sample in samples:
        path = Path(sample["path"])
        rgbs.append(load_display_image(path, config.image_size))
        tensors.append(tensor_for_image(path, config.image_size, device))
        targets.append(ClassifierOutputTarget(int(sample["pred_idx"])))

    input_tensor = torch.cat(tensors, dim=0)
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

    outputs: list[dict] = []
    for sample, rgb, grayscale_cam in zip(samples, rgbs, grayscale_cams):
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        outputs.append(
            {
                "sample": sample,
                "rgb": rgb,
                "heatmap": grayscale_cam,
                "overlay": overlay,
                "heatmap_rgb": heatmap_to_rgb(grayscale_cam),
            }
        )
    return outputs


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("inferno")
    return (cmap(heatmap)[..., :3] * 255).astype(np.uint8)


def activation_centroid(heatmap: np.ndarray) -> tuple[float, float]:
    weights = np.maximum(heatmap.astype(np.float64), 0)
    total = weights.sum()
    if total <= 0:
        return 0.5, 0.5
    rows = np.arange(weights.shape[0], dtype=np.float64)
    cols = np.arange(weights.shape[1], dtype=np.float64)
    row_centroid = float((weights.sum(axis=1) * rows).sum() / total / max(weights.shape[0] - 1, 1))
    col_centroid = float((weights.sum(axis=0) * cols).sum() / total / max(weights.shape[1] - 1, 1))
    return row_centroid, col_centroid


def centroid_region(row: float, col: float) -> str:
    vertical = "top" if row < 0.40 else "bottom" if row > 0.60 else "center"
    horizontal = "left" if col < 0.40 else "right" if col > 0.60 else "center"
    if vertical == "center" and horizontal == "center":
        return "central region"
    if vertical == "center":
        return f"center-{horizontal} region"
    if horizontal == "center":
        return f"{vertical}-center region"
    return f"{vertical}-{horizontal} region"


def short_name(class_name: str) -> str:
    return SHORT_NAMES.get(class_name, class_name.replace("_", " "))


def select_main_samples(pred_df: pd.DataFrame, class_idx: int, n_samples: int) -> list[pd.Series]:
    class_df = pred_df[pred_df["true_idx"] == class_idx].copy()
    correct_df = class_df[class_df["correct"]].sort_values("confidence", ascending=False)
    wrong_df = class_df[~class_df["correct"]].sort_values("confidence", ascending=False)

    selected: list[pd.Series] = []
    if not correct_df.empty:
        selected.append(correct_df.iloc[0])
        median_pos = len(correct_df) // 2
        selected.append(correct_df.iloc[median_pos])
    if not wrong_df.empty:
        selected.append(wrong_df.iloc[0])

    fallback_df = class_df.sort_values("confidence", ascending=False)
    for _, row in fallback_df.iterrows():
        if len(selected) >= n_samples:
            break
        if not any(row["path"] == item["path"] for item in selected):
            selected.append(row)

    return selected[:n_samples]


def draw_triptych(
    parent_ax,
    rgb_image: np.ndarray,
    overlay: np.ndarray,
    heatmap_rgb: np.ndarray,
    title: str,
    border_color: str | None = None,
) -> None:
    parent_ax.set_xticks([])
    parent_ax.set_yticks([])
    parent_ax.set_title(title, fontsize=8, pad=4)
    for spine in parent_ax.spines.values():
        spine.set_visible(border_color is not None)
        spine.set_color(border_color or "black")
        spine.set_linewidth(2.0)

    panels = [
        (rgb_image, "Original"),
        (overlay, "Grad-CAM"),
        (heatmap_rgb, "Heatmap"),
    ]
    for idx, (image, label) in enumerate(panels):
        ax = parent_ax.inset_axes([idx / 3.0 + 0.01, 0.02, 0.31, 0.88])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=6, pad=1)
        for spine in ax.spines.values():
            spine.set_visible(False)


def save_row_figure(
    row_items: list[dict],
    class_name: str,
    species_slug: str,
    n_samples: int,
) -> None:
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4), squeeze=False)
    for ax, item in zip(axes[0], row_items):
        draw_triptych(ax, item["rgb"], item["overlay"], item["heatmap_rgb"], item["title"])
    fig.suptitle(short_name(class_name), fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = GRADCAM_DIR / f"{species_slug}_row.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Saved species row: {output_path}", flush=True)


def make_main_grid(
    cam: GradCAM,
    pred_df: pd.DataFrame,
    class_names: list[str],
    config: ModelConfig,
    device: torch.device,
    n_samples: int,
) -> None:
    fig, axes = plt.subplots(len(class_names), n_samples, figsize=(15, 32), squeeze=False)
    all_centroids: dict[str, list[tuple[float, float]]] = {}

    for class_idx, class_name in enumerate(class_names):
        print(f"Processing class {class_idx + 1}/{len(class_names)}: {short_name(class_name)}...", flush=True)
        selected = select_main_samples(pred_df, class_idx, n_samples)
        cam_outputs = make_cams_batch(cam, selected, config, device)
        row_items: list[dict] = []
        centroids: list[tuple[float, float]] = []

        for sample_idx, cam_output in enumerate(cam_outputs):
            sample = cam_output["sample"]
            row, col = activation_centroid(cam_output["heatmap"])
            centroids.append((row, col))

            status = "correct" if bool(sample["correct"]) else "wrong"
            title = f"{status}, conf={float(sample['confidence']):.2f}"
            item = {
                "rgb": cam_output["rgb"],
                "overlay": cam_output["overlay"],
                "heatmap_rgb": cam_output["heatmap_rgb"],
                "title": title,
            }
            row_items.append(item)
            draw_triptych(
                axes[class_idx, sample_idx],
                cam_output["rgb"],
                cam_output["overlay"],
                cam_output["heatmap_rgb"],
                title,
            )

        for empty_idx in range(len(row_items), n_samples):
            axes[class_idx, empty_idx].axis("off")

        axes[class_idx, 0].set_ylabel(short_name(class_name), fontsize=10, rotation=0, labelpad=48, va="center")
        species_slug = class_name.replace("/", "_").replace(" ", "_")
        save_row_figure(row_items, class_name, species_slug, n_samples)
        all_centroids[class_name] = centroids

    for class_name, centroids in all_centroids.items():
        if not centroids:
            continue
        mean_row = float(np.mean([item[0] for item in centroids]))
        mean_col = float(np.mean([item[1] for item in centroids]))
        print(
            f"{short_name(class_name)}: activation centroid at "
            f"(row={mean_row:.1f}, col={mean_col:.1f}) -> {centroid_region(mean_row, mean_col)}",
            flush=True,
        )

    fig.tight_layout()
    output_path = FIGURES_DIR / "gradcam_main_grid.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Saved main Grad-CAM grid: {output_path}", flush=True)


def make_misclassified_figure(
    cam: GradCAM,
    pred_df: pd.DataFrame,
    class_names: list[str],
    config: ModelConfig,
    device: torch.device,
) -> None:
    wrong_df = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False).head(12)
    if wrong_df.empty:
        print("[Misclassified] No wrong predictions found; skipping misclassified figure.", flush=True)
        return

    samples = [row for _, row in wrong_df.iterrows()]
    cam_outputs = make_cams_batch(cam, samples, config, device)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), squeeze=False)
    for ax, cam_output in zip(axes.flatten(), cam_outputs):
        sample = cam_output["sample"]
        ax.set_xticks([])
        ax.set_yticks([])
        true_label = short_name(class_names[int(sample["true_idx"])])
        pred_label = short_name(class_names[int(sample["pred_idx"])])
        ax.set_title(f"True: {true_label} -> Pred: {pred_label} (conf={float(sample['confidence']):.2f})", fontsize=8)

        original_ax = ax.inset_axes([0.02, 0.05, 0.46, 0.82])
        gradcam_ax = ax.inset_axes([0.52, 0.05, 0.46, 0.82])
        original_ax.imshow(cam_output["rgb"])
        gradcam_ax.imshow(cam_output["overlay"])
        original_ax.set_title("Original", fontsize=7)
        gradcam_ax.set_title("Grad-CAM", fontsize=7)
        for child_ax, color in ((original_ax, "#2f9e44"), (gradcam_ax, "#c53434")):
            child_ax.set_xticks([])
            child_ax.set_yticks([])
            for spine in child_ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes.flatten()[len(cam_outputs) :]:
        ax.axis("off")

    fig.suptitle("Top-12 Misclassified Test Images", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = FIGURES_DIR / "gradcam_misclassified.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Saved misclassified Grad-CAM figure: {output_path}", flush=True)


def load_or_make_confusion_matrix(pred_df: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    cm_path = RESULTS_DIR / "confusion_matrix_raw.csv"
    if cm_path.exists():
        raw = pd.read_csv(cm_path, index_col=0)
        cm = raw.to_numpy()
        if cm.shape == (len(class_names), len(class_names)):
            print(f"[Confused Pairs] Loaded confusion matrix: {cm_path}", flush=True)
            return cm
        print(f"[Confused Pairs] Ignoring unexpected matrix shape {cm.shape}; recomputing.", flush=True)

    cm = confusion_matrix(pred_df["true_idx"], pred_df["pred_idx"], labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)
    print(f"[Confused Pairs] Saved computed confusion matrix: {cm_path}", flush=True)
    return cm


def top_confused_pairs(cm: np.ndarray, top_k: int = 3) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                pairs.append((true_idx, pred_idx, int(cm[true_idx, pred_idx])))
    return sorted(pairs, key=lambda item: item[2], reverse=True)[:top_k]


def representative_for_pair(pred_df: pd.DataFrame, true_idx: int, pred_idx: int) -> pd.Series | None:
    pair_df = pred_df[(pred_df["true_idx"] == true_idx) & (pred_df["pred_idx"] == pred_idx)]
    if not pair_df.empty:
        return pair_df.sort_values("confidence", ascending=False).iloc[0]
    correct_df = pred_df[(pred_df["true_idx"] == true_idx) & (pred_df["correct"])]
    if not correct_df.empty:
        return correct_df.sort_values("confidence", ascending=False).iloc[0]
    class_df = pred_df[pred_df["true_idx"] == true_idx]
    if not class_df.empty:
        return class_df.sort_values("confidence", ascending=False).iloc[0]
    return None


def make_confused_pairs_figure(
    cam: GradCAM,
    pred_df: pd.DataFrame,
    class_names: list[str],
    config: ModelConfig,
    device: torch.device,
) -> None:
    cm = load_or_make_confusion_matrix(pred_df, class_names)
    pairs = top_confused_pairs(cm, top_k=3)
    if not pairs:
        print("[Confused Pairs] No off-diagonal confusion found; skipping figure.", flush=True)
        return

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), squeeze=False)
    for row_idx, (a_idx, b_idx, count) in enumerate(pairs):
        samples = [
            representative_for_pair(pred_df, a_idx, b_idx),
            representative_for_pair(pred_df, b_idx, a_idx),
        ]
        present_samples = [sample for sample in samples if sample is not None]
        cam_outputs = make_cams_batch(cam, present_samples, config, device)
        output_by_path = {item["sample"]["path"]: item for item in cam_outputs}
        row_title = f"{short_name(class_names[a_idx])} -> {short_name(class_names[b_idx])} (n={count})"
        axes[row_idx, 0].set_ylabel(row_title, fontsize=10, rotation=0, labelpad=70, va="center")

        col = 0
        for sample, species_idx in zip(samples, (a_idx, b_idx)):
            if sample is None:
                axes[row_idx, col].axis("off")
                axes[row_idx, col + 1].axis("off")
                col += 2
                continue

            cam_output = output_by_path[sample["path"]]

            axes[row_idx, col].imshow(cam_output["rgb"])
            axes[row_idx, col].set_title(f"{short_name(class_names[species_idx])} original", fontsize=9)
            axes[row_idx, col + 1].imshow(cam_output["overlay"])
            axes[row_idx, col + 1].set_title(f"{short_name(class_names[species_idx])} Grad-CAM", fontsize=9)
            for ax in (axes[row_idx, col], axes[row_idx, col + 1]):
                ax.set_xticks([])
                ax.set_yticks([])
            col += 2

    for row_idx in range(len(pairs), 3):
        for ax in axes[row_idx]:
            ax.axis("off")

    fig.suptitle("Top Confused Species Pairs", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = FIGURES_DIR / "gradcam_confused_pairs.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Saved confused-pairs Grad-CAM figure: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for wood species classifier.")
    parser.add_argument("--best_model", default=BEST_MODEL, choices=sorted(MODEL_CONFIGS) + ["all"], help="Best model key, or 'all' to run all four models.")
    parser.add_argument("--n_samples", type=int, default=3, help="Images per class for the main grid.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for the initial prediction pass.")
    parser.add_argument("--resnet50_batch_size", type=int, default=None, help="Override prediction batch size when --best_model resnet50.")
    parser.add_argument("--efficientnet_b4_batch_size", type=int, default=None, help="Override prediction batch size when --best_model efficientnet_b4.")
    parser.add_argument("--convnext_tiny_batch_size", type=int, default=None, help="Override prediction batch size when --best_model convnext_tiny.")
    parser.add_argument("--vit_b16_batch_size", type=int, default=None, help="Override prediction batch size when --best_model vit_b16.")
    return parser.parse_args()


def batch_size_for_model(args: argparse.Namespace) -> int:
    model_key = args.best_model if args.best_model != "all" else "convnext_tiny"
    override = getattr(args, f"{model_key}_batch_size")
    batch_size = override if override is not None else args.batch_size
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")
    return batch_size


def batch_size_for_selected_model(args: argparse.Namespace, model_key: str) -> int:
    override = getattr(args, f"{model_key}_batch_size")
    batch_size = override if override is not None else args.batch_size
    if batch_size < 1:
        raise ValueError(f"Batch size for {model_key} must be >= 1, got {batch_size}")
    return batch_size


def run_gradcam_for_model(
    model_key: str,
    args: argparse.Namespace,
    device: torch.device,
    class_names: list[str],
    figure_dir: Path,
    gradcam_dir: Path,
) -> None:
    global FIGURES_DIR, GRADCAM_DIR

    old_figures_dir = FIGURES_DIR
    old_gradcam_dir = GRADCAM_DIR
    FIGURES_DIR = figure_dir
    GRADCAM_DIR = gradcam_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    try:
        config = MODEL_CONFIGS[model_key]
        batch_size = batch_size_for_selected_model(args, model_key)

        print("\n" + "=" * 80, flush=True)
        print(f"[Grad-CAM] Running model: {model_key} | batch_size={batch_size}", flush=True)
        print(f"[Grad-CAM] Figure dir: {FIGURES_DIR}", flush=True)
        print("=" * 80, flush=True)

        dataset, loader = build_dataset_and_loader(config.image_size, class_names, batch_size)
        model = load_model(model_key, config, len(class_names), device)
        target_layer = get_target_layer(model, model_key)
        reshape_transform = vit_reshape_transform if model_key == "vit_b16" else None

        pred_df = predict_all(model, dataset, loader, device)

        print("[Grad-CAM] Initializing GradCAM", flush=True)
        with GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
            make_main_grid(cam, pred_df, class_names, config, device, args.n_samples)
            make_misclassified_figure(cam, pred_df, class_names, config, device)
            make_confused_pairs_figure(cam, pred_df, class_names, config, device)

        print(f"[Grad-CAM] Complete for {model_key}. Figures saved under {FIGURES_DIR}", flush=True)
    finally:
        FIGURES_DIR = old_figures_dir
        GRADCAM_DIR = old_gradcam_dir


def main() -> None:
    args = parse_args()
    if args.n_samples < 1:
        raise ValueError("--n_samples must be >= 1")

    set_seed(SEED)
    device = get_device()
    label_map = load_label_map()
    class_names = [label_map[idx] for idx in sorted(label_map)]

    if args.best_model == "all":
        for model_key in MODEL_ORDER:
            try:
                run_gradcam_for_model(
                    model_key=model_key,
                    args=args,
                    device=device,
                    class_names=class_names,
                    figure_dir=FIGURES_DIR / "all_models" / model_key,
                    gradcam_dir=GRADCAM_DIR / model_key,
                )
            except Exception as exc:
                print(f"[Grad-CAM] ERROR for {model_key}: {exc}. Continuing to next model.", flush=True)
        print(f"[Grad-CAM] All models complete. Outputs saved under {FIGURES_DIR / 'all_models'}", flush=True)
    else:
        batch_size = batch_size_for_model(args)
        print(f"[Setup] Using prediction batch_size={batch_size}", flush=True)
        run_gradcam_for_model(
            model_key=args.best_model,
            args=args,
            device=device,
            class_names=class_names,
            figure_dir=FIGURES_DIR,
            gradcam_dir=GRADCAM_DIR,
        )


if __name__ == "__main__":
    main()


# nohup python gradcam_visualization.py --best_model convnext_tiny > results/gradcam_stdout.log 2>&1 &
# nohup python gradcam_visualization.py --best_model all > results/gradcam_all_stdout.log 2>&1 &
