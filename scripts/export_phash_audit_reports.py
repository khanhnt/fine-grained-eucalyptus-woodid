#!/usr/bin/env python3
"""Export cross-partition pHash audit reports for Split A and Split B.

The script compares perceptual hashes only within the same class and only
across train/validation/test partitions. Split manifests may either contain a
stored `phash` column or point to image files from which pHash values are
computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any


pd = None
Image = None
imagehash = None


def load_dependencies() -> None:
    """Import optional runtime dependencies with a clear error message."""
    global pd, Image, imagehash
    try:
        import pandas as _pd
        from PIL import Image as _Image
        import imagehash as _imagehash
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for pHash audit. Install project requirements with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    pd = _pd
    Image = _Image
    imagehash = _imagehash


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Export cross-partition perceptual-hash near-duplicate reports."
    )
    parser.add_argument(
        "--split-a",
        type=Path,
        default=root / "manifests" / "split_A_reference.csv",
        help="Path to Split A manifest CSV.",
    )
    parser.add_argument(
        "--split-b",
        type=Path,
        default=root / "manifests" / "split_B_strict.csv",
        help="Path to Split B manifest CSV.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help=(
            "Root used to resolve image paths when a split CSV has no phash column. "
            "This can be the dataset root containing raw/, or the raw/ folder itself."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "manifests" / "leakage_audit_reports",
        help="Directory for exported CSV reports.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[5, 10],
        help="pHash Hamming-distance thresholds.",
    )
    parser.add_argument(
        "--phash-size",
        type=int,
        default=8,
        help="pHash size. The default 8 gives a 64-bit perceptual hash.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print counts without writing CSV files.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    if not path.is_absolute():
        for base in (repo_root(), Path.cwd()):
            candidate = base / path
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"File not found: {path}")


def value_from_row(row: Any, column: str) -> str | None:
    if column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def image_path_candidates(row: Any, split_csv: Path, raw_root: Path | None) -> list[Path]:
    values = []
    for column in ("raw_path", "relative_path", "image_path", "path"):
        value = value_from_row(row, column)
        if value:
            values.append(value)

    bases: list[Path] = []
    if raw_root is not None:
        bases.extend([raw_root, raw_root / "raw"])
    bases.extend([split_csv.parent, repo_root(), Path.cwd()])

    candidates: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_absolute():
            candidates.append(path)
        else:
            for base in bases:
                candidates.append(base / path)
    return candidates


def resolve_image_path(row: Any, split_csv: Path, raw_root: Path | None) -> Path | None:
    for candidate in image_path_candidates(row, split_csv, raw_root):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def phash_from_image(path: Path, phash_size: int) -> Any | None:
    try:
        with Image.open(path) as image:
            return imagehash.phash(image.convert("L"), hash_size=phash_size)
    except Exception as exc:
        print(f"[WARN] Could not compute pHash for {path}: {exc}")
        return None


def sha256_from_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as exc:
        print(f"[WARN] Could not compute SHA-256 for {path}: {exc}")
        return None


def parse_phash_string(value: Any) -> Any | None:
    try:
        return imagehash.hex_to_hash(str(value).strip())
    except Exception:
        return None


def load_split(csv_path: Path, raw_root: Path | None, phash_size: int) -> Any:
    csv_path = resolve_input_path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip() for column in df.columns]

    required = {"class_name", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    hashes = []
    from_column = 0
    from_image = 0
    missing_hash = 0
    has_phash_column = "phash" in df.columns
    computed_sha256 = []

    for _, row in df.iterrows():
        hash_value = None
        image_path = None
        if has_phash_column:
            stored = value_from_row(row, "phash")
            if stored:
                hash_value = parse_phash_string(stored)
                if hash_value is not None:
                    from_column += 1

        if hash_value is None:
            image_path = resolve_image_path(row, csv_path, raw_root)
            if image_path is not None:
                hash_value = phash_from_image(image_path, phash_size)
                if hash_value is not None:
                    from_image += 1

        if "sha256" not in df.columns and image_path is not None:
            computed_sha256.append(sha256_from_file(image_path))
        else:
            computed_sha256.append(None)

        if hash_value is None:
            missing_hash += 1
        hashes.append(hash_value)

    df["_phash"] = hashes
    df["_computed_sha256"] = computed_sha256
    print(
        f"[{csv_path.name}] pHash values: {from_column} from column, "
        f"{from_image} computed from images, {missing_hash} missing"
    )

    if from_column + from_image == 0:
        raise RuntimeError(
            f"No pHash values could be obtained for {csv_path}. "
            "Provide --raw-root if the manifest does not contain a phash column."
        )
    return df


def preferred_path_column(df: Any) -> str:
    for column in ("relative_path", "raw_path", "image_path", "path", "image_id"):
        if column in df.columns:
            return column
    return df.columns[0]


def collect_phash_pairs(df: Any, thresholds: list[int]) -> dict[int, list[dict[str, Any]]]:
    path_column = preferred_path_column(df)
    pairs = {threshold: [] for threshold in thresholds}

    grouped: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        if df.at[idx, "_phash"] is None:
            continue
        grouped.setdefault(str(row["class_name"]), []).append(idx)

    for class_name, indices in grouped.items():
        for pos_a, idx_a in enumerate(indices):
            split_a = df.at[idx_a, "split"]
            hash_a = df.at[idx_a, "_phash"]
            for idx_b in indices[pos_a + 1 :]:
                split_b = df.at[idx_b, "split"]
                if split_a == split_b:
                    continue
                distance = int(hash_a - df.at[idx_b, "_phash"])
                for threshold in thresholds:
                    if distance <= threshold:
                        pairs[threshold].append(
                            {
                                "class_name": class_name,
                                "image_1": df.at[idx_a, path_column],
                                "split_1": split_a,
                                "image_2": df.at[idx_b, path_column],
                                "split_2": split_b,
                                "hamming_distance": distance,
                                "phash_1": str(hash_a),
                                "phash_2": str(df.at[idx_b, "_phash"]),
                            }
                        )

    for threshold in thresholds:
        pairs[threshold].sort(key=lambda row: (row["hamming_distance"], row["class_name"]))
    return pairs


def count_cross_split_value_overlap(df: Any, value_column: str | None) -> int | str:
    if value_column is None or value_column not in df.columns:
        return "not_available"
    tmp = df[[value_column, "split"]].dropna().copy()
    tmp[value_column] = tmp[value_column].astype(str).str.strip()
    tmp = tmp[tmp[value_column] != ""]
    if tmp.empty:
        return 0
    split_counts = tmp.groupby(value_column)["split"].nunique()
    return int((split_counts > 1).sum())


def count_filename_overlap(df: Any) -> int:
    path_column = preferred_path_column(df)
    tmp = df[[path_column, "split"]].dropna().copy()
    tmp["_filename"] = tmp[path_column].map(lambda value: Path(str(value)).name)
    split_counts = tmp.groupby("_filename")["split"].nunique()
    return int((split_counts > 1).sum())


def group_column(df: Any) -> str | None:
    for column in ("parsed_group_id", "group_id"):
        if column in df.columns:
            return column
    return None


def hash_column(df: Any) -> str | None:
    if "sha256" in df.columns:
        return "sha256"
    if "_computed_sha256" in df.columns and df["_computed_sha256"].notna().any():
        return "_computed_sha256"
    return None


def audit_summary_rows(
    df_a: Any,
    df_b: Any,
    pairs_a: dict[int, list[dict[str, Any]]],
    pairs_b: dict[int, list[dict[str, Any]]],
    thresholds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "audit_stage": "Parsed-group overlap",
            "split_A": count_cross_split_value_overlap(df_a, group_column(df_a)),
            "split_B": count_cross_split_value_overlap(df_b, group_column(df_b)),
        },
        {
            "audit_stage": "Exact cross-partition file hash",
            "split_A": count_cross_split_value_overlap(df_a, hash_column(df_a)),
            "split_B": count_cross_split_value_overlap(df_b, hash_column(df_b)),
        },
        {
            "audit_stage": "Filename overlap",
            "split_A": count_filename_overlap(df_a),
            "split_B": count_filename_overlap(df_b),
        },
    ]

    for threshold in thresholds:
        rows.append(
            {
                "audit_stage": f"pHash Hamming <= {threshold}",
                "split_A": len(pairs_a[threshold]),
                "split_B": len(pairs_b[threshold]),
            }
        )

    rows.append(
        {
            "audit_stage": "Feature nearest-neighbour",
            "split_A": "manual review",
            "split_B": "manual review",
        }
    )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_reports(
    pairs_a: dict[int, list[dict[str, Any]]],
    pairs_b: dict[int, list[dict[str, Any]]],
    summary_rows: list[dict[str, Any]],
    thresholds: list[int],
    output_dir: Path,
) -> None:
    pair_fields = [
        "class_name",
        "image_1",
        "split_1",
        "image_2",
        "split_2",
        "hamming_distance",
        "phash_1",
        "phash_2",
    ]
    written = []
    for split_tag, pairs in (("splitA", pairs_a), ("splitB", pairs_b)):
        for threshold in thresholds:
            path = output_dir / f"audit_{split_tag}_phash_pairs_d{threshold}.csv"
            write_rows(path, pairs[threshold], pair_fields)
            written.append((path, len(pairs[threshold])))

    summary_path = output_dir / "audit_summary_both_splits.csv"
    write_rows(summary_path, summary_rows, ["audit_stage", "split_A", "split_B"])

    print("\n[Export] Files written:")
    for path, count in written:
        print(f"  {path} ({count} pairs)")
    print(f"  {summary_path}")


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nAudit summary:")
    print(f"{'Audit stage':<36} {'Split A':>12} {'Split B':>12}")
    for row in summary_rows:
        print(f"{row['audit_stage']:<36} {str(row['split_A']):>12} {str(row['split_B']):>12}")


def main() -> None:
    args = parse_args()
    load_dependencies()

    thresholds = sorted(set(args.thresholds))
    df_a = load_split(args.split_a, args.raw_root, args.phash_size)
    df_b = load_split(args.split_b, args.raw_root, args.phash_size)

    pairs_a = collect_phash_pairs(df_a, thresholds)
    pairs_b = collect_phash_pairs(df_b, thresholds)
    rows = audit_summary_rows(df_a, df_b, pairs_a, pairs_b, thresholds)
    print_summary(rows)

    if not args.summary_only:
        export_reports(pairs_a, pairs_b, rows, thresholds, args.output_dir)


if __name__ == "__main__":
    main()
