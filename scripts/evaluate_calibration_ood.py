#!/usr/bin/env python3
"""Calibration and OOD/unseen-species evaluation for a trained classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from paper_utils import (
    MODEL_CONFIGS,
    ManifestImageDataset,
    UnlabeledImageDataset,
    build_model,
    eval_transform,
    infer_classes_from_split,
    is_image,
    load_checkpoint_state,
    load_split_file,
    make_loader,
    resolve_device,
    save_json,
)


def collect_ood_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing OOD folder: {root}")
    paths = sorted(path for path in root.rglob("*") if is_image(path))
    if not paths:
        raise FileNotFoundError(f"No OOD images found under: {root}")
    return paths


@torch.no_grad()
def infer_known(model, loader, device, use_amp: bool) -> pd.DataFrame:
    rows = []
    model.eval()
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        energy_np = (-torch.logsumexp(logits.float(), dim=1)).detach().cpu().numpy()
        probs_np = probs.detach().cpu().float().numpy()
        pred = probs_np.argmax(axis=1)
        max_prob = probs_np.max(axis=1)
        entropy = -(probs_np * np.log(np.clip(probs_np, 1e-12, 1))).sum(axis=1)
        energy = energy_np
        for i, path in enumerate(paths):
            rows.append(
                {
                    "path": path,
                    "label": int(labels[i]),
                    "pred": int(pred[i]),
                    "correct": int(pred[i]) == int(labels[i]),
                    "max_softmax": float(max_prob[i]),
                    "entropy": float(entropy[i]),
                    "energy": float(energy[i]),
                    **{f"prob_{j}": float(probs_np[i, j]) for j in range(probs_np.shape[1])},
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def infer_ood(model, loader, device, use_amp: bool) -> pd.DataFrame:
    rows = []
    model.eval()
    for images, paths in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        energy_np = (-torch.logsumexp(logits.float(), dim=1)).detach().cpu().numpy()
        probs_np = probs.detach().cpu().float().numpy()
        pred = probs_np.argmax(axis=1)
        max_prob = probs_np.max(axis=1)
        entropy = -(probs_np * np.log(np.clip(probs_np, 1e-12, 1))).sum(axis=1)
        energy = energy_np
        for i, path in enumerate(paths):
            rows.append(
                {
                    "path": path,
                    "pred": int(pred[i]),
                    "max_softmax": float(max_prob[i]),
                    "entropy": float(entropy[i]),
                    "energy": float(energy[i]),
                    **{f"prob_{j}": float(probs_np[i, j]) for j in range(probs_np.shape[1])},
                }
            )
    return pd.DataFrame(rows)


def calibration_bins(known_df: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, dict]:
    conf = known_df["max_softmax"].to_numpy()
    correct = known_df["correct"].astype(int).to_numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            acc = 0.0
            avg_conf = 0.0
            gap = 0.0
        else:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            gap = abs(acc - avg_conf)
        prop = n / max(len(conf), 1)
        ece += prop * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": idx,
                "bin_lower": lo,
                "bin_upper": hi,
                "n": n,
                "accuracy": acc,
                "confidence": avg_conf,
                "gap": gap,
            }
        )

    y = known_df["label"].to_numpy()
    probs = known_df[[col for col in known_df.columns if col.startswith("prob_")]].to_numpy()
    nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))
    metrics = {
        "nll": float(nll),
        "brier_score": float(brier),
        "ece": float(ece),
        "mce": float(mce),
        "accuracy": float(accuracy_score(y, known_df["pred"].to_numpy())),
    }
    return pd.DataFrame(rows), metrics


def plot_reliability(bin_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(
        bin_df["confidence"],
        bin_df["accuracy"],
        width=0.08,
        alpha=0.75,
        color="#4f7f6f",
        label="Observed",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def ood_metric_from_score(score_known: np.ndarray, score_ood: np.ndarray, score_name: str) -> dict:
    y_true = np.concatenate([np.ones_like(score_known), np.zeros_like(score_ood)])
    scores = np.concatenate([score_known, score_ood])
    try:
        auroc = roc_auc_score(y_true, scores)
        aupr_in = average_precision_score(y_true, scores)
        aupr_out = average_precision_score(1 - y_true, -scores)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        valid = np.where(tpr >= 0.95)[0]
        fpr95 = float(fpr[valid[0]]) if len(valid) else 1.0
    except ValueError:
        auroc = float("nan")
        aupr_in = float("nan")
        aupr_out = float("nan")
        fpr95 = float("nan")

    candidate_thresholds = np.unique(scores)
    best_acc = 0.0
    best_threshold = float(candidate_thresholds[0])
    for threshold in candidate_thresholds:
        known_ok = score_known >= threshold
        ood_ok = score_ood < threshold
        acc = (known_ok.sum() + ood_ok.sum()) / (len(score_known) + len(score_ood))
        if acc > best_acc:
            best_acc = float(acc)
            best_threshold = float(threshold)
    return {
        "score": score_name,
        "auroc": float(auroc),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out),
        "fpr_at_95_tpr": fpr95,
        "best_detection_accuracy": best_acc,
        "best_threshold": best_threshold,
    }


def plot_histograms(known_df: pd.DataFrame, ood_df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("max_softmax", "Confidence", "confidence_histogram_known_vs_ood.png"),
        ("entropy", "Entropy", "entropy_histogram_known_vs_ood.png"),
        ("energy", "Energy", "energy_histogram_known_vs_ood.png"),
    ]
    for column, title, filename in specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(known_df[column], bins=30, alpha=0.65, label="Known test", density=True)
        ax.hist(ood_df[column], bins=30, alpha=0.65, label="OOD / unseen", density=True)
        ax.set_title(f"{title}: Known vs OOD")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)


def plot_ood_curves(known_df: pd.DataFrame, ood_df: pd.DataFrame, out_dir: Path) -> None:
    y_true = np.concatenate([np.ones(len(known_df)), np.zeros(len(ood_df))])
    scores = np.concatenate([known_df["max_softmax"], ood_df["max_softmax"]])
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD ROC Curve (MSP knownness)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ood_roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("OOD PR Curve (In-distribution positive)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ood_pr_curve.png", dpi=300)
    plt.close(fig)


def threshold_analysis(known_df: pd.DataFrame, ood_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        known_accept = int((known_df["max_softmax"] >= threshold).sum())
        known_reject = int((known_df["max_softmax"] < threshold).sum())
        ood_accept = int((ood_df["max_softmax"] >= threshold).sum())
        ood_reject = int((ood_df["max_softmax"] < threshold).sum())
        rows.append(
            {
                "threshold": threshold,
                "known_samples_accepted": known_accept,
                "known_samples_rejected": known_reject,
                "ood_samples_accepted_as_known": ood_accept,
                "ood_samples_rejected": ood_reject,
                "known_acceptance_rate": known_accept / max(len(known_df), 1),
                "ood_rejection_rate": ood_reject / max(len(ood_df), 1),
            }
        )
    return pd.DataFrame(rows)


def forced_distribution(ood_df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    df = ood_df.copy()
    df["predicted_class"] = df["pred"].map(lambda idx: class_names[int(idx)])
    out = (
        df.groupby("predicted_class")
        .agg(
            n_ood_images=("predicted_class", "size"),
            mean_confidence=("max_softmax", "mean"),
            std_confidence=("max_softmax", "std"),
            min_confidence=("max_softmax", "min"),
            median_confidence=("max_softmax", "median"),
            max_confidence=("max_softmax", "max"),
        )
        .reset_index()
    )
    out["percentage"] = out["n_ood_images"] / len(df) * 100
    return out.sort_values("n_ood_images", ascending=False)


def save_tex_tables(out_dir: Path, calibration_metrics: dict, ood_metrics: pd.DataFrame, threshold_df: pd.DataFrame, forced_df: pd.DataFrame) -> None:
    pd.DataFrame([calibration_metrics]).to_latex(
        out_dir / "calibration_metrics.tex",
        index=False,
        float_format="%.4f",
        caption="Calibration metrics on the known held-out test split.",
        label="tab:calibration_metrics",
    )
    ood_metrics.to_latex(
        out_dir / "ood_metrics.tex",
        index=False,
        float_format="%.4f",
        caption="OOD detection metrics using unseen E. globulus images.",
        label="tab:ood_metrics",
    )
    threshold_df.to_latex(
        out_dir / "ood_threshold_analysis.tex",
        index=False,
        float_format="%.4f",
        caption="Confidence-threshold rejection analysis for known and unseen samples.",
        label="tab:ood_threshold_analysis",
    )
    forced_df.to_latex(
        out_dir / "ood_forced_prediction_distribution.tex",
        index=False,
        float_format="%.4f",
        caption="Forced closed-set prediction distribution for unseen E. globulus samples.",
        label="tab:ood_forced_predictions",
    )


def write_report(out_dir: Path, calibration_metrics: dict, ood_metrics: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    msp = ood_metrics[ood_metrics["score"] == "max_softmax"].iloc[0]
    best_threshold = threshold_df.sort_values("ood_rejection_rate", ascending=False).iloc[0]
    lines = [
        "# Calibration and OOD Evaluation",
        "",
        "## Calibration on Known Test Split",
        "",
        f"- Accuracy: {calibration_metrics['accuracy'] * 100:.2f}%",
        f"- NLL: {calibration_metrics['nll']:.4f}",
        f"- Brier score: {calibration_metrics['brier_score']:.4f}",
        f"- ECE: {calibration_metrics['ece']:.4f}",
        f"- MCE: {calibration_metrics['mce']:.4f}",
        "",
        "## OOD Detection with Unseen E. globulus",
        "",
        f"- MSP AUROC: {msp['auroc']:.4f}",
        f"- MSP AUPR-In: {msp['aupr_in']:.4f}",
        f"- MSP AUPR-Out: {msp['aupr_out']:.4f}",
        f"- MSP FPR@95TPR: {msp['fpr_at_95_tpr']:.4f}",
        "",
        "## Interpretation",
        "",
        (
            "The E. globulus stress test should not be interpreted as closed-set accuracy. "
            "It characterizes how the closed-set classifier behaves when exposed to an unseen congeneric species. "
            "Softmax confidence should be interpreted carefully: unseen Eucalyptus samples may still be assigned "
            "to one of the known labels with non-negligible confidence."
        ),
        "",
        f"Highest OOD rejection among requested thresholds occurred at threshold {best_threshold['threshold']:.2f}, "
        f"with OOD rejection rate {best_threshold['ood_rejection_rate'] * 100:.2f}%.",
        "",
    ]
    (out_dir / "calibration_ood_report.md").write_text("\n".join(lines), encoding="utf-8")
    snippet = (
        "The E. globulus stress test should not be interpreted as closed-set accuracy. Instead, it characterizes "
        "how the closed-set classifier behaves when exposed to an unseen congeneric species. Calibration and OOD "
        "metrics indicate whether max-softmax confidence, entropy, or energy provide useful signals for rejecting "
        "samples outside the known label space; however, conclusions are limited by the use of a single unseen species.\n"
    )
    (out_dir / "calibration_ood_manuscript_snippet.md").write_text(snippet, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibration and OOD behavior.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--known-split-file", type=Path, required=True)
    parser.add_argument("--ood-data-root", type=Path, required=True)
    parser.add_argument("--model", default="resnet50", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--out-dir", type=Path, default=Path("results/calibration_ood"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.out_dir / "config.json", vars(args))
    device = resolve_device(args.device)

    split_df = load_split_file(args.known_split_file)
    class_names = infer_classes_from_split(split_df)
    known_test = split_df[split_df["split"] == "test"].reset_index(drop=True)
    if known_test.empty:
        raise ValueError("Known split file has no rows with split == 'test'.")

    known_loader = make_loader(
        ManifestImageDataset(known_test, eval_transform(args.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    ood_paths = collect_ood_images(args.ood_data_root)
    ood_loader = DataLoader(
        UnlabeledImageDataset(ood_paths, eval_transform(args.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.model, num_classes=len(class_names), pretrained=False, image_size=args.img_size).to(device)
    load_checkpoint_state(model, args.checkpoint, device)

    known_scores = infer_known(model, known_loader, device, args.use_amp)
    ood_scores = infer_ood(model, ood_loader, device, args.use_amp)
    known_scores["predicted_class"] = known_scores["pred"].map(lambda idx: class_names[int(idx)])
    ood_scores["predicted_class"] = ood_scores["pred"].map(lambda idx: class_names[int(idx)])
    known_scores.to_csv(args.out_dir / "known_confidence_scores.csv", index=False)
    ood_scores.to_csv(args.out_dir / "ood_confidence_scores.csv", index=False)

    bin_df, cal_metrics = calibration_bins(known_scores, args.bins)
    bin_df.to_csv(args.out_dir / "calibration_bins.csv", index=False)
    pd.DataFrame([cal_metrics]).to_csv(args.out_dir / "calibration_metrics.csv", index=False)
    plot_reliability(bin_df, args.out_dir / "reliability_diagram.png")

    plot_histograms(known_scores, ood_scores, args.out_dir)
    ood_rows = [
        ood_metric_from_score(known_scores["max_softmax"].to_numpy(), ood_scores["max_softmax"].to_numpy(), "max_softmax"),
        ood_metric_from_score((-known_scores["entropy"]).to_numpy(), (-ood_scores["entropy"]).to_numpy(), "negative_entropy"),
        ood_metric_from_score((-known_scores["energy"]).to_numpy(), (-ood_scores["energy"]).to_numpy(), "negative_energy"),
    ]
    ood_metrics = pd.DataFrame(ood_rows)
    ood_metrics.to_csv(args.out_dir / "ood_metrics.csv", index=False)
    plot_ood_curves(known_scores, ood_scores, args.out_dir)

    threshold_df = threshold_analysis(known_scores, ood_scores)
    threshold_df.to_csv(args.out_dir / "ood_threshold_analysis.csv", index=False)
    forced_df = forced_distribution(ood_scores, class_names)
    forced_df.to_csv(args.out_dir / "ood_forced_prediction_distribution.csv", index=False)

    save_tex_tables(args.out_dir, cal_metrics, ood_metrics, threshold_df, forced_df)
    write_report(args.out_dir, cal_metrics, ood_metrics, threshold_df)
    print(f"[Done] Calibration/OOD outputs saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
