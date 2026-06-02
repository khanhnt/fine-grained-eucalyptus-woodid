#!/usr/bin/env python3
"""Leakage audit for an existing manuscript split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from paper_utils import (
    ManifestImageDataset,
    collect_images_from_root,
    eval_transform,
    file_sha256,
    load_split_file,
    make_loader,
    records_to_dataframe,
    resolve_device,
    save_contact_sheet,
    save_json,
)

try:
    import imagehash
    from PIL import Image, ImageOps
except ModuleNotFoundError:
    imagehash = None
    Image = None
    ImageOps = None


def export_split_from_data_root(data_root: Path, out_path: Path) -> pd.DataFrame:
    data_root = data_root.resolve()
    records = collect_images_from_root(data_root)
    df = records_to_dataframe(records)
    if {"train", "val", "test"}.issubset({p.name for p in data_root.iterdir() if p.is_dir()}):
        split_values = []
        for path_str in df["image_path"]:
            path = Path(path_str)
            rel = path.relative_to(data_root)
            split_values.append(rel.parts[0])
        df["split"] = split_values
    else:
        raise ValueError("--data-root must contain train/val/test folders to export an existing split.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[Export] Wrote current split file: {out_path}", flush=True)
    return df


def split_counts(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_counts = split_df.groupby("split").size().reset_index(name="n_images")
    group_counts = split_df.groupby("split")["group_id"].nunique().reset_index(name="n_groups")
    return image_counts, group_counts


def group_overlap_audit(split_df: pd.DataFrame, out_dir: Path) -> dict:
    rows = []
    overlaps = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        left_groups = set(split_df.loc[split_df["split"] == left, "group_id"])
        right_groups = set(split_df.loc[split_df["split"] == right, "group_id"])
        overlap = sorted(left_groups & right_groups)
        overlaps[f"{left}_vs_{right}"] = overlap
        rows.append({"pair": f"{left}_vs_{right}", "n_overlap_groups": len(overlap), "groups": ";".join(overlap[:100])})
    pd.DataFrame(rows).to_csv(out_dir / "leakage_group_audit.csv", index=False)
    payload = {"passed": all(len(v) == 0 for v in overlaps.values()), "overlaps": overlaps}
    save_json(out_dir / "leakage_group_audit.json", payload)
    return payload


def exact_hash_audit(split_df: pd.DataFrame, out_dir: Path) -> dict:
    rows = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="SHA256 hashes"):
        rows.append({**row.to_dict(), "sha256": file_sha256(Path(row["image_path"]))})
    hash_df = pd.DataFrame(rows)

    duplicate_rows = []
    for sha, group in hash_df.groupby("sha256"):
        if group["split"].nunique() > 1:
            duplicate_rows.extend(group.to_dict("records"))

    out = pd.DataFrame(duplicate_rows)
    out.to_csv(out_dir / "leakage_exact_hash_audit.csv", index=False)
    summary = {
        "n_cross_split_duplicate_hashes": int(out["sha256"].nunique()) if not out.empty else 0,
        "n_duplicate_rows": int(len(out)),
        "passed": out.empty,
    }
    save_json(out_dir / "leakage_exact_hash_summary.json", summary)
    return summary


def filename_audit(split_df: pd.DataFrame, out_dir: Path) -> dict:
    df = split_df.copy()
    df["filename"] = df["image_path"].map(lambda p: Path(p).name)
    df["normalized_filename"] = df["filename"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    rows = []
    for column in ["filename", "normalized_filename"]:
        for name, group in df.groupby(column):
            if group["split"].nunique() > 1:
                for item in group.to_dict("records"):
                    item["audit_type"] = column
                    rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "leakage_filename_audit.csv", index=False)
    return {
        "n_cross_split_filename_duplicates": int(out[out.get("audit_type", pd.Series(dtype=str)) == "filename"]["filename"].nunique()) if not out.empty else 0,
        "n_rows": int(len(out)),
        "passed": out.empty,
    }


def compute_phash(path: Path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return imagehash.phash(image)


def phash_audit(split_df: pd.DataFrame, out_dir: Path, thresholds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if imagehash is None:
        print("[WARN] imagehash is not installed; perceptual-hash audit skipped.", flush=True)
        empty_pairs = pd.DataFrame()
        empty_summary = pd.DataFrame([{"threshold": t, "n_pairs": np.nan, "note": "imagehash not installed"} for t in thresholds])
        empty_pairs.to_csv(out_dir / "leakage_phash_pairs.csv", index=False)
        empty_summary.to_csv(out_dir / "leakage_phash_summary.csv", index=False)
        return empty_pairs, empty_summary

    rows = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="pHash"):
        rows.append({**row.to_dict(), "phash": compute_phash(Path(row["image_path"]))})
    phash_df = pd.DataFrame(rows)
    max_threshold = max(thresholds)
    pair_rows = []

    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in split_pairs:
        left_df = phash_df[phash_df["split"] == left]
        right_df = phash_df[phash_df["split"] == right]
        for _, q in tqdm(left_df.iterrows(), total=len(left_df), desc=f"pHash {left}-{right}", leave=False):
            qhash = q["phash"]
            for _, n in right_df.iterrows():
                dist = int(qhash - n["phash"])
                if dist <= max_threshold:
                    pair_rows.append(
                        {
                            "query_path": q["image_path"],
                            "query_split": q["split"],
                            "query_class": q["class_name"],
                            "query_group": q["group_id"],
                            "neighbor_path": n["image_path"],
                            "neighbor_split": n["split"],
                            "neighbor_class": n["class_name"],
                            "neighbor_group": n["group_id"],
                            "phash_distance": dist,
                        }
                    )

    pairs = pd.DataFrame(pair_rows).sort_values("phash_distance") if pair_rows else pd.DataFrame()
    pairs.to_csv(out_dir / "leakage_phash_pairs.csv", index=False)
    summary_rows = []
    for threshold in thresholds:
        n_pairs = int((pairs["phash_distance"] <= threshold).sum()) if not pairs.empty else 0
        summary_rows.append({"threshold": threshold, "n_pairs": n_pairs})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "leakage_phash_summary.csv", index=False)
    return pairs, summary


def build_feature_extractor(device: torch.device):
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(split_df: pd.DataFrame, batch_size: int, num_workers: int, device: torch.device) -> tuple[np.ndarray, list[str]]:
    dataset = ManifestImageDataset(split_df, eval_transform(224))
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = build_feature_extractor(device)
    feats = []
    paths = []
    for images, _, batch_paths in tqdm(loader, desc="ResNet-50 features"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(images)
        out = torch.nn.functional.normalize(out.float(), dim=1)
        feats.append(out.cpu().numpy())
        paths.extend(batch_paths)
    return np.concatenate(feats, axis=0), paths


def feature_nn_audit(split_df: pd.DataFrame, out_dir: Path, top_k: int, batch_size: int, num_workers: int, device: torch.device) -> pd.DataFrame:
    feats, paths = extract_features(split_df, batch_size=batch_size, num_workers=num_workers, device=device)
    path_to_idx = {path: idx for idx, path in enumerate(paths)}
    meta = split_df.set_index("image_path")

    train_paths = split_df.loc[split_df["split"] == "train", "image_path"].tolist()
    query_paths = split_df.loc[split_df["split"].isin(["val", "test"]), "image_path"].tolist()
    train_indices = [path_to_idx[path] for path in train_paths]
    query_indices = [path_to_idx[path] for path in query_paths]
    train_feats = feats[train_indices]

    rows = []
    for q_path, q_idx in tqdm(list(zip(query_paths, query_indices)), desc="Feature nearest neighbors"):
        sims = train_feats @ feats[q_idx]
        top_indices = np.argsort(-sims)[:top_k]
        q = meta.loc[q_path]
        for rank, local_idx in enumerate(top_indices, start=1):
            n_path = train_paths[int(local_idx)]
            n = meta.loc[n_path]
            rows.append(
                {
                    "query_path": q_path,
                    "query_split": q["split"],
                    "query_class": q["class_name"],
                    "query_group": q["group_id"],
                    "neighbor_path": n_path,
                    "neighbor_split": n["split"],
                    "neighbor_class": n["class_name"],
                    "neighbor_group": n["group_id"],
                    "cosine_similarity": float(sims[int(local_idx)]),
                    "rank": rank,
                }
            )
    nn_df = pd.DataFrame(rows)
    nn_df.to_csv(out_dir / "leakage_feature_nn_pairs.csv", index=False)
    return nn_df


def make_contact_sheets(phash_pairs: pd.DataFrame, nn_pairs: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    contact_dir = out_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    if not phash_pairs.empty:
        for _, row in phash_pairs.sort_values("phash_distance").head(top_n).iterrows():
            title = f"pHash d={row['phash_distance']} | {row['query_split']}:{row['query_class']} -> {row['neighbor_split']}:{row['neighbor_class']}"
            candidates.append(("phash", row["query_path"], row["neighbor_path"], title))
    if not nn_pairs.empty:
        top_nn = nn_pairs[nn_pairs["rank"] == 1].sort_values("cosine_similarity", ascending=False).head(top_n)
        for _, row in top_nn.iterrows():
            title = f"NN sim={row['cosine_similarity']:.4f} | {row['query_split']}:{row['query_class']} -> {row['neighbor_split']}:{row['neighbor_class']}"
            candidates.append(("feature", row["query_path"], row["neighbor_path"], title))

    for idx, (kind, query, neighbor, title) in enumerate(candidates, start=1):
        save_contact_sheet(Path(query), Path(neighbor), title, contact_dir / f"{idx:03d}_{kind}.png")


def save_summary_tex(out_dir: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "leakage_audit_summary.csv", index=False)
    tex = df.to_latex(
        index=False,
        escape=False,
        caption="Leakage audit summary for the acquisition/specimen-group-disjoint split.",
        label="tab:leakage_audit",
    )
    (out_dir / "leakage_audit_summary.tex").write_text(tex, encoding="utf-8")


def write_markdown_report(
    out_dir: Path,
    split_df: pd.DataFrame,
    group_payload: dict,
    exact_summary: dict,
    filename_summary: dict,
    phash_summary: pd.DataFrame,
    nn_df: pd.DataFrame,
) -> None:
    image_counts, group_counts = split_counts(split_df)
    max_nn = float(nn_df["cosine_similarity"].max()) if not nn_df.empty else float("nan")
    phash_pairs = int(phash_summary["n_pairs"].fillna(0).max()) if not phash_summary.empty else 0
    lines = [
        "# Leakage Audit Report",
        "",
        "## Split Size",
        "",
        image_counts.to_markdown(index=False),
        "",
        "## Group Counts",
        "",
        group_counts.to_markdown(index=False),
        "",
        "## Audit Summary",
        "",
        f"- Group overlap passed: `{group_payload['passed']}`",
        f"- Cross-split exact duplicate hashes: `{exact_summary['n_cross_split_duplicate_hashes']}`",
        f"- Cross-split filename duplicate rows: `{filename_summary['n_rows']}`",
        f"- Perceptual-hash candidate pairs at selected thresholds: `{phash_pairs}`",
        f"- Maximum feature-nearest-neighbor cosine similarity: `{max_nn:.4f}`",
        "",
        "## Interpretation",
        "",
        (
            "The audit checks group identifiers, exact file hashes, filenames, perceptual hashes, "
            "and ImageNet ResNet-50 feature nearest neighbors across train/validation/test partitions. "
            "The appropriate manuscript wording should be conservative: no exact duplicate or group-overlap "
            "evidence was found only if the corresponding tables report zero findings. Perceptual-hash and "
            "feature-nearest-neighbor pairs should be inspected visually using the contact sheets before "
            "claiming absence of near-duplicate leakage."
        ),
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "leakage_audit_report.md").write_text(report, encoding="utf-8")
    snippet = (
        "Leakage audits evaluated shared specimen/group identifiers, exact file hashes, filename overlap, "
        "perceptual-hash near duplicates, and feature-nearest-neighbor similarity across train, validation, "
        "and test partitions. These audits should be interpreted as evidence against obvious leakage under "
        "the performed checks, rather than as proof that all possible acquisition biases are absent.\n"
    )
    (out_dir / "leakage_audit_manuscript_snippet.md").write_text(snippet, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage audits for a split.csv file.")
    parser.add_argument("--split-file", type=Path, default=None, help="CSV with image_path,label,class_name,group_id,split.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional train/val/test root to export split.csv if needed.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/leakage_audit"))
    parser.add_argument("--phash-thresholds", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.split_file is None:
        if args.data_root is None:
            raise ValueError("Provide --split-file, or provide --data-root to export a split file.")
        split_df = export_split_from_data_root(args.data_root, args.out_dir / "current_split_export.csv")
    elif args.split_file.exists():
        split_df = load_split_file(args.split_file)
    elif args.data_root is not None:
        split_df = export_split_from_data_root(args.data_root, args.split_file)
    else:
        raise FileNotFoundError(f"Missing split file: {args.split_file}")

    device = resolve_device(args.device)
    image_counts, group_counts = split_counts(split_df)
    image_counts.to_csv(args.out_dir / "split_image_counts.csv", index=False)
    group_counts.to_csv(args.out_dir / "split_group_counts.csv", index=False)

    group_payload = group_overlap_audit(split_df, args.out_dir)
    exact_summary = exact_hash_audit(split_df, args.out_dir)
    filename_summary = filename_audit(split_df, args.out_dir)
    phash_pairs, phash_summary = phash_audit(split_df, args.out_dir, args.phash_thresholds)
    nn_df = feature_nn_audit(split_df, args.out_dir, args.top_k, args.batch_size, args.num_workers, device)
    make_contact_sheets(phash_pairs, nn_df, args.out_dir, top_n=args.top_k)

    summary_rows = [
        {"audit": "group_overlap", "finding_count": sum(len(v) for v in group_payload["overlaps"].values()), "passed": group_payload["passed"]},
        {"audit": "exact_hash", "finding_count": exact_summary["n_cross_split_duplicate_hashes"], "passed": exact_summary["passed"]},
        {"audit": "filename", "finding_count": filename_summary["n_rows"], "passed": filename_summary["passed"]},
        {"audit": "phash", "finding_count": int(phash_summary["n_pairs"].fillna(0).max()) if not phash_summary.empty else 0, "passed": (phash_pairs.empty if isinstance(phash_pairs, pd.DataFrame) else True)},
        {"audit": "feature_nn", "finding_count": len(nn_df), "passed": "manual_review_required"},
    ]
    save_summary_tex(args.out_dir, summary_rows)
    write_markdown_report(args.out_dir, split_df, group_payload, exact_summary, filename_summary, phash_summary, nn_df)
    print(f"[Done] Leakage audit outputs saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
