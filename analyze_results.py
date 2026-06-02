#!/usr/bin/env python3
"""Analyze benchmark results and generate figures for the wood classifier study."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---
BEST_MODEL = "convnext_tiny"
SEED = 42
DEFAULT_BATCH_SIZE = 256
NUM_WORKERS = 8
USE_AMP = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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

LOG_FILES = {
    "ResNet50": RESULTS_DIR / "logs" / "ResNet50_seed42_log.csv",
    "EfficientNetB4": RESULTS_DIR / "logs" / "EfficientNetB4_seed42_log.csv",
    "ConvNeXtTiny": RESULTS_DIR / "logs" / "ConvNeXtTiny_seed42_log.csv",
    "ViTB16": RESULTS_DIR / "logs" / "ViTB16_seed42_log.csv",
}


class FixedClassImageFolder(datasets.ImageFolder):
    """ImageFolder with a fixed class_to_idx mapping."""

    def __init__(self, root: Path, classes: list[str] | None = None, **kwargs):
        self.fixed_classes = classes
        super().__init__(root=str(root), **kwargs)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        if self.fixed_classes is None:
            return super().find_classes(directory)
        return self.fixed_classes, {class_name: idx for idx, class_name in enumerate(self.fixed_classes)}


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_analysis() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Setup] Using GPU: {gpu_name} | VRAM: {total_vram_gb:.2f} GB", flush=True)
        return device
    print("[Setup] CUDA not available; using CPU for analysis.", flush=True)
    return torch.device("cpu")


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


def build_test_loader(image_size: int, expected_classes: list[str], batch_size: int) -> DataLoader:
    test_dir = DATA_DIR / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test split: {test_dir}")

    dataset = FixedClassImageFolder(test_dir, classes=expected_classes, transform=eval_transform(image_size))
    if dataset.classes != expected_classes:
        raise ValueError(
            "ImageFolder class order does not match results/label_map.json. "
            f"ImageFolder={dataset.classes}, label_map={expected_classes}"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


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
        raise RuntimeError("Could not create ViT-B/16 model with the installed timm version.")

    raise ValueError(f"Unknown model family: {config.family}")


def resolve_checkpoint(best_model: str, config: ModelConfig) -> Path:
    requested_path = CKPT_DIR / f"{best_model}_best.pth"
    if requested_path.exists():
        return requested_path

    fallback_path = CKPT_DIR / f"{config.display_name}_best.pth"
    if fallback_path.exists():
        print(
            f"[Checkpoint] Requested {requested_path} not found; using benchmark fallback {fallback_path}",
            flush=True,
        )
        return fallback_path

    raise FileNotFoundError(
        f"Best checkpoint not found. Expected {requested_path}. "
        f"Also checked compatibility fallback {fallback_path}."
    )


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_best_model(best_model: str, config: ModelConfig, num_classes: int, device: torch.device) -> nn.Module:
    checkpoint_path = resolve_checkpoint(best_model, config)
    print(f"[Checkpoint] Loading best model: {checkpoint_path}", flush=True)

    model = build_model(config, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_test_set(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    all_labels: list[int] = []
    all_preds: list[int] = []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Evaluating test set", leave=False)

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=USE_AMP and device.type == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.asarray(all_labels), np.asarray(all_preds)


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: list[str]) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
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
    output_path = FIGURES_DIR / "confusion_matrix_8x8.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Saved confusion matrix: {output_path}", flush=True)


def save_per_class_f1(labels: np.ndarray, preds: np.ndarray, class_names: list[str]) -> None:
    f1_values = f1_score(labels, preds, labels=list(range(len(class_names))), average=None, zero_division=0)
    report_df = pd.DataFrame({"class_name": class_names, "f1": f1_values})
    report_df.to_csv(RESULTS_DIR / "per_class_f1.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, f1_values, color="#4f7f6f")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1")
    ax.grid(axis="x", alpha=0.25)

    for idx, value in enumerate(f1_values):
        ax.text(min(value + 0.015, 0.98), idx, f"{value:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    output_path = FIGURES_DIR / "per_class_f1.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Saved per-class F1 chart: {output_path}", flush=True)


def save_classification_report(labels: np.ndarray, preds: np.ndarray, class_names: list[str]) -> None:
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    output_path = RESULTS_DIR / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(output_path)
    print(f"[Metrics] Saved classification report: {output_path}", flush=True)


def save_learning_curves() -> None:
    found_any = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, log_path in LOG_FILES.items():
        if not log_path.exists():
            print(f"[Learning Curves] Missing log, skipping {model_name}: {log_path}", flush=True)
            continue

        df = pd.read_csv(log_path)
        if df.empty or "epoch" not in df:
            print(f"[Learning Curves] Empty or invalid log, skipping {model_name}: {log_path}", flush=True)
            continue

        found_any = True
        if "train_loss" in df:
            axes[0].plot(df["epoch"], df["train_loss"], label=f"{model_name} Train")
        if "val_loss" in df:
            axes[0].plot(df["epoch"], df["val_loss"], linestyle="--", label=f"{model_name} Val")
        if "val_f1_macro" in df:
            axes[1].plot(df["epoch"], df["val_f1_macro"], label=model_name)

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path = FIGURES_DIR / "learning_curves_all_models.png"
    if found_any:
        fig.savefig(output_path, dpi=300)
        print(f"[Figure] Saved learning curves: {output_path}", flush=True)
    else:
        print("[Learning Curves] No logs found; no learning curve figure saved.", flush=True)
    plt.close(fig)


def extract_features_for_tsne(model: nn.Module, config: ModelConfig, images: torch.Tensor) -> torch.Tensor:
    if config.family == "torchvision_resnet50":
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    if config.family == "torchvision_convnext_tiny":
        x = model.features(images)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    if hasattr(model, "forward_features"):
        x = model.forward_features(images)
        if hasattr(model, "forward_head"):
            try:
                x = model.forward_head(x, pre_logits=True)
            except TypeError:
                x = model.forward_head(x)
        if x.ndim == 3:
            x = x[:, 0]
        elif x.ndim == 4:
            x = x.mean(dim=(2, 3))
        return torch.flatten(x, 1)

    raise ValueError(f"Feature extraction is not implemented for {config.family}")


@torch.no_grad()
def collect_features(model: nn.Module, config: ModelConfig, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    print("Extracting features for t-SNE...", flush=True)
    features: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Extracting features for t-SNE", leave=True)

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=USE_AMP and device.type == "cuda"):
            batch_features = extract_features_for_tsne(model, config, images)
        features.append(batch_features.detach().float().cpu().numpy())
        labels_out.append(labels.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels_out, axis=0)


def save_tsne_plot(features: np.ndarray, labels: np.ndarray, class_names: list[str]) -> None:
    if len(features) < 3:
        print("[t-SNE] Need at least 3 samples; skipping t-SNE figure.", flush=True)
        return

    perplexity = min(30, max(2, (len(features) - 1) // 3))
    print(f"[t-SNE] Running sklearn TSNE with n_jobs=-1, perplexity={perplexity}", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=SEED,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(features)

    tsne_df = pd.DataFrame(
        {
            "tsne_1": coords[:, 0],
            "tsne_2": coords[:, 1],
            "label": labels,
            "class_name": [class_names[idx] for idx in labels],
        }
    )
    tsne_df.to_csv(RESULTS_DIR / "tsne_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=16,
            alpha=0.78,
            color=cmap(class_idx % 10),
            label=class_name,
            edgecolors="none",
        )

    ax.set_title("t-SNE of Test-Set Features")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7, markerscale=1.6, loc="best")
    fig.tight_layout()

    output_path = FIGURES_DIR / "tsne_features.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Saved t-SNE plot: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze wood species classification benchmark results.")
    parser.add_argument("--best_model", default=BEST_MODEL, choices=sorted(MODEL_CONFIGS), help="Best model key.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for evaluation and feature extraction.")
    parser.add_argument("--resnet50_batch_size", type=int, default=None, help="Override batch size when --best_model resnet50.")
    parser.add_argument("--efficientnet_b4_batch_size", type=int, default=None, help="Override batch size when --best_model efficientnet_b4.")
    parser.add_argument("--convnext_tiny_batch_size", type=int, default=None, help="Override batch size when --best_model convnext_tiny.")
    parser.add_argument("--vit_b16_batch_size", type=int, default=None, help="Override batch size when --best_model vit_b16.")
    return parser.parse_args()


def batch_size_for_model(args: argparse.Namespace) -> int:
    override = getattr(args, f"{args.best_model}_batch_size")
    batch_size = override if override is not None else args.batch_size
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")
    return batch_size


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = device_for_analysis()

    config = MODEL_CONFIGS[args.best_model]
    batch_size = batch_size_for_model(args)
    label_map = load_label_map()
    class_names = [label_map[idx] for idx in sorted(label_map)]

    print("[Analysis] Plotting learning curves from all available logs.", flush=True)
    save_learning_curves()

    print(f"[Analysis] Loading test set at image_size={config.image_size}, batch_size={batch_size}", flush=True)
    test_loader = build_test_loader(config.image_size, class_names, batch_size)
    model = load_best_model(args.best_model, config, num_classes=len(class_names), device=device)

    print("[Analysis] Evaluating best model on test set.", flush=True)
    labels, preds = predict_test_set(model, test_loader, device)
    save_classification_report(labels, preds, class_names)
    save_confusion_matrix(labels, preds, class_names)
    save_per_class_f1(labels, preds, class_names)

    features, feature_labels = collect_features(model, config, test_loader, device)
    save_tsne_plot(features, feature_labels, class_names)

    print(f"[Analysis] Complete. Figures saved to {FIGURES_DIR}", flush=True)


if __name__ == "__main__":
    main()


# python analyze_results.py --best_model convnext_tiny
