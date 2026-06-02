#!/usr/bin/env python3
"""Repeated acquisition/specimen-group-disjoint split evaluation.

This script runs robustness experiments using the same model families,
ImageNet normalization, augmentation policy, checkpoint dictionary style, and
specimen-group parser as the benchmark utilities.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from paper_utils import (
    MODEL_CONFIGS,
    ManifestImageDataset,
    build_model,
    classification_metrics,
    collect_images_from_root,
    eval_transform,
    file_md5,
    infer_classes_from_split,
    load_checkpoint_state,
    make_loader,
    mean_std_summary,
    per_class_metrics,
    predict_manifest,
    records_to_dataframe,
    resolve_device,
    save_confusion_outputs,
    save_json,
    set_seed,
    train_transform,
)


METRIC_COLS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def deduplicate_exact_hashes(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    hash_to_indices: dict[str, list[int]] = {}
    for idx, path_str in enumerate(df["image_path"]):
        h = file_md5(Path(path_str))
        hash_to_indices.setdefault(h, []).append(idx)

    keep_indices = []
    for h, indices in hash_to_indices.items():
        keep_indices.append(indices[0])
        if len(indices) > 1:
            for idx in indices:
                row = df.iloc[idx].to_dict()
                row["md5"] = h
                rows.append(row)

    if rows:
        duplicate_path = out_dir / "exact_hash_duplicates_removed.csv"
        pd.DataFrame(rows).to_csv(duplicate_path, index=False)
        print(f"[Dedup] Removed {len(df) - len(keep_indices)} exact duplicate files. Report: {duplicate_path}", flush=True)
    else:
        print("[Dedup] No exact duplicate files found.", flush=True)

    return df.iloc[sorted(keep_indices)].reset_index(drop=True)


def split_group_disjoint(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    split_frames = []
    warnings = []

    for class_name, class_df in df.groupby("class_name", sort=True):
        class_df = class_df.reset_index(drop=True)
        groups = class_df["group_id"].tolist()
        unique_groups = sorted(set(groups))

        if len(unique_groups) < 3:
            warnings.append(
                {
                    "class_name": class_name,
                    "warning": f"Only {len(unique_groups)} groups; all images assigned to train.",
                }
            )
            class_df["split"] = "train"
            split_frames.append(class_df)
            continue

        indices = np.arange(len(class_df))
        gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
        train_idx, temp_idx = next(gss_train.split(indices, groups=groups))

        temp_df = class_df.iloc[temp_idx].reset_index(drop=True)
        temp_groups = temp_df["group_id"].tolist()
        if len(set(temp_groups)) < 2:
            class_df["split"] = "train"
            class_df.loc[temp_idx, "split"] = "val"
            split_frames.append(class_df)
            warnings.append({"class_name": class_name, "warning": "Temp split has <2 groups; test is empty."})
            continue

        temp_indices = np.arange(len(temp_df))
        gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_rel_idx, test_rel_idx = next(gss_val_test.split(temp_indices, groups=temp_groups))

        class_df["split"] = ""
        class_df.loc[train_idx, "split"] = "train"
        original_temp_indices = class_df.iloc[temp_idx].index.to_numpy()
        class_df.loc[original_temp_indices[val_rel_idx], "split"] = "val"
        class_df.loc[original_temp_indices[test_rel_idx], "split"] = "test"
        split_frames.append(class_df)

    out = pd.concat(split_frames, ignore_index=True)
    if warnings:
        print("[WARN] Split warnings:")
        for item in warnings:
            print(f"  {item['class_name']}: {item['warning']}", flush=True)
    return out[["image_path", "label", "class_name", "group_id", "split"]]


def group_overlap_report(split_df: pd.DataFrame, out_dir: Path) -> dict:
    split_groups = {split: set(split_df.loc[split_df["split"] == split, "group_id"]) for split in ["train", "val", "test"]}
    rows = []
    overlaps = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(split_groups[left] & split_groups[right])
        overlaps[f"{left}_vs_{right}"] = overlap
        rows.append({"pair": f"{left}_vs_{right}", "n_overlap_groups": len(overlap), "groups": ";".join(overlap[:50])})

    pd.DataFrame(rows).to_csv(out_dir / "group_overlap_checks.csv", index=False)
    payload = {
        "n_groups": {split: len(groups) for split, groups in split_groups.items()},
        "overlaps": overlaps,
        "passed": all(len(groups) == 0 for groups in overlaps.values()),
    }
    save_json(out_dir / "group_overlap_checks.json", payload)
    if not payload["passed"]:
        raise AssertionError(f"Group overlap detected: {payload['overlaps']}")
    return payload


def split_support_report(split_df: pd.DataFrame, out_dir: Path) -> None:
    support = (
        split_df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["split", "class_name"])
    )
    group_support = (
        split_df.groupby(["split", "class_name"])["group_id"]
        .nunique()
        .reset_index(name="n_groups")
        .sort_values(["split", "class_name"])
    )
    merged = support.merge(group_support, on=["split", "class_name"], how="outer").fillna(0)
    merged.to_csv(out_dir / "split_class_support.csv", index=False)
    small = merged[(merged["split"].isin(["val", "test"])) & (merged["n_images"] < 3)]
    if not small.empty:
        print("[WARN] Very small validation/test support detected:")
        print(small.to_string(index=False), flush=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1)


def evaluate_loss_and_metrics(model, loader, criterion, device, class_names, use_amp: bool) -> tuple[float, dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels_device)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            labels_all.extend(labels.numpy().tolist())
            preds_all.extend(preds.tolist())
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
    y_true = np.array(labels_all)
    y_pred = np.array(preds_all)
    metrics = classification_metrics(y_true, y_pred, class_names) if len(y_true) else {m: 0.0 for m in METRIC_COLS}
    return total_loss / max(total, 1), metrics, y_true, y_pred


def run_seed(args: argparse.Namespace, base_df: pd.DataFrame, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    seed_dir = args.out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model}_best.pth"

    config = {
        "seed": seed,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "data_root": str(args.data_root),
        "out_dir": str(seed_dir),
        "checkpoint": str(ckpt_path),
    }
    save_json(seed_dir / "config.json", config)

    split_df = split_group_disjoint(base_df, seed)
    split_df.to_csv(seed_dir / "split.csv", index=False)
    overlap_payload = group_overlap_report(split_df, seed_dir)
    split_support_report(split_df, seed_dir)

    class_names = infer_classes_from_split(split_df)
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)

    train_ds = ManifestImageDataset(train_df, train_transform(args.img_size))
    val_ds = ManifestImageDataset(val_df, eval_transform(args.img_size))
    test_ds = ManifestImageDataset(test_df, eval_transform(args.img_size))
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, seed=seed)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, num_classes=len(class_names), pretrained=True, image_size=args.img_size).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    best_epoch = 0
    best_val_macro_f1 = -1.0
    train_log_path = seed_dir / "train_log.csv"
    train_logs = []

    if ckpt_path.exists() and not args.overwrite:
        if args.no_resume:
            raise FileExistsError(
                f"Checkpoint already exists and --no-resume was set: {ckpt_path}. "
                "Use --overwrite to retrain."
            )
        print(f"[Seed {seed}] Existing checkpoint found; loading without overwrite: {ckpt_path}", flush=True)
        ckpt = load_checkpoint_state(model, ckpt_path, device)
        best_epoch = int(ckpt.get("epoch", 0))
        best_val_macro_f1 = float(ckpt.get("best_f1", ckpt.get("best_val_macro_f1", -1.0)))
    else:
        if ckpt_path.exists() and args.overwrite:
            print(f"[Seed {seed}] Overwriting checkpoint: {ckpt_path}", flush=True)
        with train_log_path.open("w", newline="", encoding="utf-8", buffering=1) as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "lr", "elapsed_sec"])
            for epoch in range(1, args.epochs + 1):
                start = time.perf_counter()
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.use_amp)
                val_loss, val_metrics, _, _ = evaluate_loss_and_metrics(model, val_loader, criterion, device, class_names, args.use_amp)
                scheduler.step()
                elapsed = int(time.perf_counter() - start)
                row = [
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics["macro_f1"],
                    val_metrics["accuracy"],
                    optimizer.param_groups[0]["lr"],
                    elapsed,
                ]
                writer.writerow(row)
                file.flush()
                train_logs.append(row)
                print(
                    f"[Seed {seed}] Ep {epoch:02d}/{args.epochs} | "
                    f"Loss {train_loss:.3f} | ValF1 {val_metrics['macro_f1']:.4f} | {elapsed}s",
                    flush=True,
                )
                if val_metrics["macro_f1"] > best_val_macro_f1:
                    best_val_macro_f1 = val_metrics["macro_f1"]
                    best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_f1": best_val_macro_f1,
                            "best_val_macro_f1": best_val_macro_f1,
                            "config": config,
                            "train_logs": train_logs,
                        },
                        ckpt_path,
                    )

    load_checkpoint_state(model, ckpt_path, device)
    _, test_metrics, y_true, y_pred = evaluate_loss_and_metrics(model, test_loader, criterion, device, class_names, args.use_amp)
    save_confusion_outputs(y_true, y_pred, class_names, seed_dir / "confusion_matrix.csv", seed_dir / "confusion_matrix.png")
    per_class_metrics(y_true, y_pred, class_names).to_csv(seed_dir / "per_class_metrics.csv", index=False)

    metric_row = {
        "seed": seed,
        "model": args.model,
        "train_images": len(train_df),
        "val_images": len(val_df),
        "test_images": len(test_df),
        "train_groups": train_df["group_id"].nunique(),
        "val_groups": val_df["group_id"].nunique(),
        "test_groups": test_df["group_id"].nunique(),
        **test_metrics,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
    }
    pd.DataFrame([metric_row]).to_csv(seed_dir / "metrics.csv", index=False)
    save_json(
        seed_dir / "best_checkpoint_info.json",
        {
            "checkpoint_path": str(ckpt_path),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "group_overlap_passed": overlap_payload["passed"],
        },
    )
    return metric_row


def save_repeated_summary(all_results: pd.DataFrame, out_dir: Path, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_dir / "repeated_split_all_results.csv", index=False)
    summary = mean_std_summary(all_results, METRIC_COLS)
    summary.to_csv(out_dir / "repeated_split_summary.csv", index=False)

    tex_df = summary.copy()
    tex_df["mean ± std"] = tex_df.apply(lambda r: f"{r['mean'] * 100:.2f} $\\pm$ {r['std'] * 100:.2f}", axis=1)
    tex_df["min"] = tex_df["min"].map(lambda x: f"{x * 100:.2f}")
    tex_df["max"] = tex_df["max"].map(lambda x: f"{x * 100:.2f}")
    tex_df = tex_df[["metric", "mean ± std", "min", "max"]]
    tex = tex_df.to_latex(
        index=False,
        escape=False,
        caption=f"Repeated acquisition/specimen-group-disjoint split evaluation for {model_name}.",
        label="tab:repeated_split",
    )
    (out_dir / "repeated_split_summary.tex").write_text(tex, encoding="utf-8")

    macro = summary.loc[summary["metric"] == "macro_f1"].iloc[0]
    report = [
        "# Repeated Split Evaluation",
        "",
        f"Model: `{model_name}`",
        f"Seeds: {', '.join(map(str, all_results['seed'].tolist()))}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Manuscript-ready interpretation",
        "",
        (
            f"Across {len(all_results)} acquisition/specimen-group-disjoint splits, "
            f"{model_name} achieved a mean macro-F1 of {macro['mean'] * 100:.2f} ± "
            f"{macro['std'] * 100:.2f} percentage points. This repeated-split analysis "
            "tests whether the strong performance is stable across random group-disjoint partitions "
            "rather than being limited to a single favorable split."
        ),
        "",
    ]
    (out_dir / "repeated_split_report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "repeated_split_manuscript_snippet.md").write_text(report[-2] + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated specimen-group-disjoint split experiments.")
    parser.add_argument("--data-root", type=Path, required=True, help="Raw class-folder root or existing train/val/test root.")
    parser.add_argument("--model", default="resnet50", choices=sorted(MODEL_CONFIGS), help="Model to evaluate.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2025, 3407], help="Split/training seeds.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--out-dir", type=Path, default=Path("results/repeated_split"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/repeated_split"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--resume", action="store_true", help="Compatibility flag; existing checkpoints are reused by default.")
    parser.add_argument("--no-resume", action="store_true", help="Compatibility flag; use --overwrite to force retraining.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing repeated-split checkpoints.")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deduplicate-exact-hashes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[Setup] device={device}", flush=True)

    records = collect_images_from_root(args.data_root)
    base_df = records_to_dataframe(records)
    if args.deduplicate_exact_hashes:
        base_df = deduplicate_exact_hashes(base_df, args.out_dir)

    all_rows = []
    for seed in args.seeds:
        print("\n" + "=" * 72, flush=True)
        print(f"[Seed {seed}] Starting repeated split evaluation", flush=True)
        row = run_seed(args, base_df, seed, device)
        all_rows.append(row)

    all_results = pd.DataFrame(all_rows)
    save_repeated_summary(all_results, args.out_dir, args.model)
    print(f"[Done] Repeated split outputs saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
