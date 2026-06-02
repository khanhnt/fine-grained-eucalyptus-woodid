#!/usr/bin/env python3
"""Materialize a split manifest into an ImageFolder train/val/test directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def class_to_folder(class_name: str) -> str:
    return class_name.strip().replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy or symlink split manifest images into ImageFolder folders.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Root folder containing raw class folders.")
    parser.add_argument("--split-csv", type=Path, required=True, help="Split manifest with relative_path,class_name,split.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output ImageFolder root.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for label_map and split export.")
    parser.add_argument("--copy", action="store_true", help="Copy images. Default is symlink.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output-dir before materializing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw root: {args.raw_root}")
    if not args.split_csv.is_file():
        raise FileNotFoundError(f"Missing split CSV: {args.split_csv}")

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    required = {"relative_path", "class_name", "class_index", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing required columns: {sorted(missing)}")

    exported_rows = []
    errors = []
    classes = (
        df[["class_index", "class_name"]]
        .drop_duplicates()
        .sort_values("class_index")
        .itertuples(index=False)
    )
    label_map = {int(row.class_index): class_to_folder(str(row.class_name)) for row in classes}

    for row in df.itertuples(index=False):
        src = args.raw_root / row.relative_path
        split = str(row.split)
        class_folder = class_to_folder(str(row.class_name))
        dst_dir = args.output_dir / split / class_folder
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(row.relative_path).name

        if not src.is_file():
            errors.append({"relative_path": row.relative_path, "error": "missing_source"})
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())

        exported_rows.append(
            {
                "image_path": str(dst.resolve()),
                "label": int(row.class_index),
                "class_name": class_folder,
                "group_id": getattr(row, "parsed_group_id", ""),
                "split": split,
            }
        )

    with (args.results_dir / "label_map.json").open("w", encoding="utf-8") as file:
        json.dump({str(k): v for k, v in label_map.items()}, file, indent=2)

    split_out = args.results_dir / f"current_split_{args.split_csv.stem}.csv"
    pd.DataFrame(exported_rows).to_csv(split_out, index=False)

    if errors:
        error_path = args.results_dir / f"materialize_errors_{args.split_csv.stem}.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        raise RuntimeError(f"Materialized with {len(errors)} missing files. See {error_path}")

    counts = pd.DataFrame(exported_rows).groupby(["split", "class_name"]).size().reset_index(name="n_images")
    counts_path = args.results_dir / f"materialized_counts_{args.split_csv.stem}.csv"
    counts.to_csv(counts_path, index=False)

    print(f"[Done] Materialized {len(exported_rows)} images into {args.output_dir}")
    print(f"[Done] Label map: {args.results_dir / 'label_map.json'}")
    print(f"[Done] Split file for audits/OOD: {split_out}")
    print(f"[Done] Counts: {counts_path}")


if __name__ == "__main__":
    main()
