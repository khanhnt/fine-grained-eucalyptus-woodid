#!/usr/bin/env python3
"""Train and benchmark wood species classifiers on an RTX 3090 GPU server."""

from __future__ import annotations

import argparse
import os
import csv
import gc
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ━━━ PATHS ━━━
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "results" / "logs"


# ━━━ GLOBAL CONFIG ━━━
SEED = int(os.environ.get("WOOD_SEED", "42"))
CURRENT_SEED = SEED
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
NUM_WORKERS = 8
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FixedClassImageFolder(datasets.ImageFolder):
    """ImageFolder with a fixed class_to_idx mapping across split folders."""

    def __init__(self, root: Path, classes: list[str] | None = None, **kwargs):
        self.fixed_classes = classes
        super().__init__(root=str(root), **kwargs)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        if self.fixed_classes is None:
            return super().find_classes(directory)
        return self.fixed_classes, {class_name: idx for idx, class_name in enumerate(self.fixed_classes)}


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    family: str
    image_size: int
    batch_size: int


MODEL_CONFIGS = {
    "resnet50": ModelConfig(
        key="resnet50",
        display_name="ResNet50",
        family="torchvision_resnet50",
        image_size=224,
        batch_size=256,
    ),
    "efficientnet_b4": ModelConfig(
        key="efficientnet_b4",
        display_name="EfficientNetB4",
        family="timm_efficientnet_b4",
        image_size=224,
        batch_size=256,
    ),
    "convnext_tiny": ModelConfig(
        key="convnext_tiny",
        display_name="ConvNeXtTiny",
        family="torchvision_convnext_tiny",
        image_size=224,
        batch_size=256,
    ),
    "vit_b16": ModelConfig(
        key="vit_b16",
        display_name="ViTB16",
        family="timm_vit_b16_384_in21k",
        image_size=384,
        batch_size=256,
    ),
}

MODEL_ORDER = ["resnet50", "efficientnet_b4", "convnext_tiny", "vit_b16"]


def log_prefix(config: ModelConfig) -> str:
    return f"[{config.display_name}]"


def set_seed(seed: int = SEED) -> None:
    """Make training as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    """Create all required output directories."""
    for directory in (DATA_DIR, RESULTS_DIR, CKPT_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def assert_dataset_layout() -> None:
    """Fail early if the ImageFolder split layout is missing."""
    required = [DATA_DIR / "train", DATA_DIR / "val", DATA_DIR / "test"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing required split folders: {missing}")


def print_gpu_info() -> torch.device:
    """Assert CUDA availability and print the active GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark, but torch.cuda.is_available() is False.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[Setup] GPU: {gpu_name} | VRAM: {total_vram_gb:.2f} GB", flush=True)
    return device


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Create train and eval transforms for the requested input size."""
    train_tfms = transforms.Compose(
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
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers deterministically."""
    worker_seed = CURRENT_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(image_size: int, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create train, val, and test ImageFolder DataLoaders."""
    train_tfms, eval_tfms = build_transforms(image_size)

    train_ds = FixedClassImageFolder(DATA_DIR / "train", transform=train_tfms)
    class_names = train_ds.classes
    val_ds = FixedClassImageFolder(DATA_DIR / "val", classes=class_names, transform=eval_tfms)
    test_ds = FixedClassImageFolder(DATA_DIR / "test", classes=class_names, transform=eval_tfms)

    generator = torch.Generator()
    generator.manual_seed(CURRENT_SEED)

    pin_memory = True
    persistent_workers = NUM_WORKERS > 0
    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": persistent_workers,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_names


def build_model(config: ModelConfig, num_classes: int) -> nn.Module:
    """Build one pretrained classifier with its final layer replaced."""
    if config.family == "torchvision_resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if config.family == "torchvision_convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if config.family == "timm_efficientnet_b4":
        import timm

        return timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)

    if config.family == "timm_vit_b16_384_in21k":
        import timm

        # timm renamed some pretrained tags across releases. Try 21K variants
        # from newest to oldest so the script remains usable on common servers.
        candidate_names = [
            "vit_base_patch16_384.augreg_in21k",
            "vit_base_patch16_384.orig_in21k",
            "vit_base_patch16_384.augreg_in21k_ft_in1k",
            "vit_base_patch16_384.orig_in21k_ft_in1k",
            "vit_base_patch16_384",
        ]
        last_error: Exception | None = None
        for model_name in candidate_names:
            try:
                return timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=num_classes,
                    img_size=384,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not create a ViT-B/16 ImageNet-21K timm model: {last_error}")

    raise ValueError(f"Unknown model family: {config.family}")


def count_params_m(model: nn.Module) -> float:
    """Return trainable plus frozen parameter count in millions."""
    return sum(param.numel() for param in model.parameters()) / 1e6


def checkpoint_path(config: ModelConfig, suffix: str) -> Path:
    return CKPT_DIR / f"{config.display_name}_seed{CURRENT_SEED}_{suffix}.pth"


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    best_f1: float,
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    val_f1s: list[float],
) -> None:
    """Save a full training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_f1s": val_f1s,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW | None = None,
    scheduler: CosineAnnealingLR | None = None,
    device: torch.device | str = "cuda",
) -> dict:
    """Load a checkpoint into the model and optionally optimizer/scheduler."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def append_csv_row(path: Path, header: list[str], row: list) -> None:
    """Append a row and flush immediately so tail -f works over SSH."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8", buffering=1) as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
        file.flush()


def init_epoch_log(path: Path, resume: bool) -> None:
    """Create the per-epoch log, preserving it only for resumed runs."""
    if path.exists() and resume:
        return
    with path.open("w", newline="", encoding="utf-8", buffering=1) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1_macro", "lr", "elapsed_sec"])
        file.flush()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    """Run one AMP training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, float, list[int], list[int]]:
    """Evaluate loss, accuracy, precision, recall, and macro F1."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, accuracy, precision, recall, f1_macro, all_labels, all_preds


@torch.no_grad()
def benchmark_inference_ms(
    model: nn.Module,
    image_size: int,
    device: torch.device,
    runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float]:
    """Measure batch=1 inference latency and peak allocated VRAM."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        with autocast():
            _ = model(dummy)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        with autocast():
            _ = model(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    inference_ms = elapsed / runs * 1000.0
    vram_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return inference_ms, vram_peak_gb


def train_model_once(
    config: ModelConfig,
    epochs: int,
    batch_size: int,
    resume: bool,
    device: torch.device,
) -> dict:
    """Train one model with a fixed batch size, then test the best checkpoint."""
    prefix = log_prefix(config)
    print(f"{prefix} Preparing data loaders at {config.image_size}px, batch={batch_size}", flush=True)
    train_loader, val_loader, test_loader, class_names = build_loaders(config.image_size, batch_size)
    num_classes = len(class_names)

    print(f"{prefix} Building pretrained model for {num_classes} classes", flush=True)
    model = build_model(config, num_classes).to(device)
    params_m = count_params_m(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_path = checkpoint_path(config, "best")
    start_epoch = 1
    best_f1 = -1.0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    val_f1s: list[float] = []
    resume_loaded = False

    if best_path.exists() and resume:
        print(f"{prefix} --resume set; loading checkpoint: {best_path}", flush=True)
        ckpt = load_checkpoint(best_path, model, optimizer, scheduler, device)
        start_epoch = int(ckpt["epoch"]) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        train_losses = list(ckpt.get("train_losses", []))
        val_losses = list(ckpt.get("val_losses", []))
        val_accs = list(ckpt.get("val_accs", []))
        val_f1s = list(ckpt.get("val_f1s", []))
        resume_loaded = True
    elif best_path.exists():
        print(f"{prefix} Existing best checkpoint found. Starting fresh because --resume was not set.", flush=True)
    elif resume:
        print(f"{prefix} --resume set but no checkpoint exists at {best_path}; starting fresh.", flush=True)

    log_path = LOG_DIR / f"{config.display_name}_seed{CURRENT_SEED}_log.csv"
    init_epoch_log(log_path, resume=resume_loaded)

    if start_epoch > epochs:
        print(f"{prefix} Checkpoint epoch is already >= requested epochs. Skipping training.", flush=True)
    else:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_acc, _, _, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            elapsed_sec = int(time.perf_counter() - epoch_start)
            lr = optimizer.param_groups[0]["lr"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

            append_csv_row(
                log_path,
                ["epoch", "train_loss", "val_loss", "val_acc", "val_f1_macro", "lr", "elapsed_sec"],
                [epoch, train_loss, val_loss, val_acc, val_f1, lr, elapsed_sec],
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_checkpoint(
                    best_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_f1,
                    train_losses,
                    val_losses,
                    val_accs,
                    val_f1s,
                )
                print(f"{prefix} New best ValF1={best_f1:.4f}; saved {best_path}", flush=True)

            if epoch % 10 == 0:
                periodic_path = checkpoint_path(config, f"epoch{epoch}")
                save_checkpoint(
                    periodic_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_f1,
                    train_losses,
                    val_losses,
                    val_accs,
                    val_f1s,
                )
                print(f"{prefix} Saved checkpoint: {periodic_path}", flush=True)

            reserved_gb = torch.cuda.memory_reserved(device) / 1e9
            print(
                f"{prefix} Ep {epoch:02d}/{epochs} | Loss {train_loss:.3f} | "
                f"ValF1 {val_f1:.3f} | {elapsed_sec}s",
                flush=True,
            )
            print(f"{prefix} GPU memory reserved: {reserved_gb:.2f} GB", flush=True)

    if not best_path.exists():
        print(f"{prefix} No best checkpoint was created; saving current model as best fallback.", flush=True)
        save_checkpoint(
            best_path,
            max(start_epoch - 1, 0),
            model,
            optimizer,
            scheduler,
            best_f1,
            train_losses,
            val_losses,
            val_accs,
            val_f1s,
        )

    print(f"{prefix} Loading best checkpoint for test evaluation", flush=True)
    best_ckpt = load_checkpoint(best_path, model, device=device)
    best_val_f1 = float(best_ckpt.get("best_f1", best_f1))

    test_loss, test_acc, test_precision, test_recall, test_f1, _, _ = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    inference_ms, vram_peak_gb = benchmark_inference_ms(model, config.image_size, device)

    append_csv_row(
        RESULTS_DIR / "test_results.csv",
        ["seed", "model", "accuracy", "precision", "recall", "f1_macro", "inference_ms", "params_M", "vram_peak_GB"],
        [
            CURRENT_SEED,
            config.display_name,
            test_acc,
            test_precision,
            test_recall,
            test_f1,
            inference_ms,
            params_m,
            vram_peak_gb,
        ],
    )

    print(
        f"{prefix} Test | Loss {test_loss:.3f} | Acc {test_acc:.4f} | "
        f"F1 {test_f1:.4f} | Infer {inference_ms:.2f} ms/img | VRAM peak {vram_peak_gb:.2f} GB",
        flush=True,
    )

    return {
        "Seed": CURRENT_SEED,
        "Model": config.display_name,
        "Val F1": best_val_f1,
        "Test Acc": test_acc,
        "Test F1": test_f1,
        "Infer(ms)": inference_ms,
        "Params(M)": params_m,
        "VRAM(GB)": vram_peak_gb,
    }


def clear_cuda() -> None:
    """Release as much GPU memory as possible between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True for common CUDA out-of-memory RuntimeErrors."""
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def train_model_with_retry(
    config: ModelConfig,
    epochs: int,
    resume: bool,
    device: torch.device,
    initial_batch_size: int,
) -> dict | None:
    """Train one model, halving batch size and retrying once after CUDA OOM."""
    batch_size = initial_batch_size
    for attempt in range(2):
        try:
            return train_model_once(config, epochs, batch_size, resume, device)
        except RuntimeError as exc:
            if is_cuda_oom(exc) and attempt == 0:
                old_batch = batch_size
                batch_size = max(1, batch_size // 2)
                print(
                    f"{log_prefix(config)} WARNING: CUDA OOM at batch={old_batch}; "
                    f"retrying once with batch={batch_size}",
                    flush=True,
                )
                clear_cuda()
                continue
            raise
        finally:
            clear_cuda()
    return None


def write_summary(rows: list[dict]) -> None:
    """Print and save the final benchmark summary table."""
    if not rows:
        print("[Summary] No completed model runs to summarize.", flush=True)
        return

    summary = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFinal benchmark summary:", flush=True)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"[Summary] Saved: {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train benchmark classifiers for wood species images.")
    parser.add_argument(
        "--model",
        choices=MODEL_ORDER,
        default=None,
        help="Train one model only. Default: train all models sequentially.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/{model}_best.pth if present.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Override number of epochs.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for this run.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Default batch size for all models.")
    parser.add_argument("--resnet50_batch_size", type=int, default=None, help="Override batch size for ResNet-50.")
    parser.add_argument("--efficientnet_b4_batch_size", type=int, default=None, help="Override batch size for EfficientNet-B4.")
    parser.add_argument("--convnext_tiny_batch_size", type=int, default=None, help="Override batch size for ConvNeXt-Tiny.")
    parser.add_argument("--vit_b16_batch_size", type=int, default=None, help="Override batch size for ViT-B/16.")
    return parser.parse_args()


def batch_size_for_model(args: argparse.Namespace, model_key: str) -> int:
    override = getattr(args, f"{model_key}_batch_size")
    batch_size = override if override is not None else args.batch_size
    if batch_size < 1:
        raise ValueError(f"Batch size for {model_key} must be >= 1, got {batch_size}")
    return batch_size


def main() -> None:
    args = parse_args()
    global CURRENT_SEED
    CURRENT_SEED = args.seed
    ensure_dirs()
    assert_dataset_layout()
    set_seed(args.seed)
    device = print_gpu_info()

    selected_models = [args.model] if args.model else MODEL_ORDER
    summary_rows: list[dict] = []

    for model_key in selected_models:
        config = MODEL_CONFIGS[model_key]
        batch_size = batch_size_for_model(args, model_key)
        print(f"\n{log_prefix(config)} Starting benchmark run", flush=True)
        print(f"{log_prefix(config)} Using batch_size={batch_size}", flush=True)
        try:
            row = train_model_with_retry(config, args.epochs, args.resume, device, batch_size)
            if row is not None:
                summary_rows.append(row)
        except Exception as exc:
            error_path = LOG_DIR / f"{config.display_name}_error.log"
            with error_path.open("a", encoding="utf-8") as file:
                file.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} {config.display_name} failed\n")
                file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                file.write("\n")
            print(f"{log_prefix(config)} ERROR: {exc}", flush=True)
            print(f"{log_prefix(config)} Error details saved to {error_path}; continuing.", flush=True)
            clear_cuda()
            continue

    write_summary(summary_rows)


if __name__ == "__main__":
    main()


# How to run on remote:
# nohup python train_benchmark.py > results/train_stdout.log 2>&1 &
# Monitor: tail -f results/logs/ResNet50_seed42_log.csv  # or seed2025/seed3407
# Monitor GPU: watch -n5 nvidia-smi
