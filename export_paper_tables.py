#!/usr/bin/env python3
"""Export publication-ready LaTeX and Markdown tables from experiment CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "paper_tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---
BEST_MODEL = "convnext_tiny"

TEST_RESULTS_CSV = RESULTS_DIR / "test_results.csv"
PER_CLASS_CSV = RESULTS_DIR / "per_class_metrics.csv"
ABLATION_RESOLUTION_CSV = RESULTS_DIR / "ablation_resolution.csv"
ABLATION_AUGMENTATION_CSV = RESULTS_DIR / "ablation_augmentation.csv"
ABLATION_FINETUNING_CSV = RESULTS_DIR / "ablation_finetuning.csv"
SPLIT_STATS_CSV = RESULTS_DIR / "split_stats.csv"

MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "efficientnet_b4": "EfficientNet-B4",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b16": "ViT-B/16",
    "ResNet50": "ResNet-50",
    "EfficientNetB4": "EfficientNet-B4",
    "ConvNeXtTiny": "ConvNeXt-Tiny",
    "ViTB16": "ViT-B/16",
}

MODEL_KEY_ALIASES = {
    "resnet50": "resnet50",
    "ResNet50": "resnet50",
    "ResNet-50": "resnet50",
    "efficientnet_b4": "efficientnet_b4",
    "EfficientNetB4": "efficientnet_b4",
    "EfficientNet-B4": "efficientnet_b4",
    "convnext_tiny": "convnext_tiny",
    "ConvNeXtTiny": "convnext_tiny",
    "ConvNeXt-Tiny": "convnext_tiny",
    "vit_b16": "vit_b16",
    "ViTB16": "vit_b16",
    "ViT-B/16": "vit_b16",
}

SPECIES_LATEX = {
    "Eucalyptus_grandis": r"\textit{E. grandis}",
    "Eucalyptus_microcorys": r"\textit{E. microcorys}",
    "Eucalyptus_saligna": r"\textit{E. saligna}",
    "Eucalyptus_deglupta": r"\textit{E. deglupta}",
    "Eucalyptus_daglupta": r"\textit{E. daglupta}",
    "Eucalyptus_diversicolor": r"\textit{E. diversicolor}",
    "Eucalyptus_cladocalyx": r"\textit{E. cladocalyx}",
    "Eucalyptus_camaldulensis": r"\textit{E. camaldulensis}",
    "Eucalyptus_camandulensis": r"\textit{E. camandulensis}",
    "Syzygium_hemisphericum": r"\textit{S. hemisphericum}",
}

SPECIES_MD = {
    "Eucalyptus_grandis": "E. grandis",
    "Eucalyptus_microcorys": "E. microcorys",
    "Eucalyptus_saligna": "E. saligna",
    "Eucalyptus_deglupta": "E. deglupta",
    "Eucalyptus_daglupta": "E. daglupta",
    "Eucalyptus_diversicolor": "E. diversicolor",
    "Eucalyptus_cladocalyx": "E. cladocalyx",
    "Eucalyptus_camaldulensis": "E. camaldulensis",
    "Eucalyptus_camandulensis": "E. camandulensis",
    "Syzygium_hemisphericum": "S. hemisphericum",
}

TEST_REQUIRED = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1_macro",
    "inference_ms",
    "params_M",
    "vram_peak_GB",
]

PER_CLASS_REQUIRED = ["class", "precision", "recall", "f1", "support"]

ABLATION_REQUIRED = [
    "ablation_group",
    "setting",
    "val_f1",
    "test_acc",
    "test_f1",
    "training_time_min",
    "inference_ms",
    "vram_peak_GB",
]


def print_saved(path: Path) -> None:
    print(f"✓ Saved {path.name}", flush=True)


def print_missing(path: Path) -> None:
    print(f"✗ Missing: {path}", flush=True)


def read_csv_checked(path: Path, required_columns: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        print_missing(path)
        return None

    df = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print(f"✗ Missing columns in {path.name}: {missing_columns}", flush=True)
        return None
    return df


def model_key(value: str) -> str:
    return MODEL_KEY_ALIASES.get(str(value), str(value))


def display_model(value: str) -> str:
    key = model_key(value)
    return MODEL_DISPLAY.get(key, MODEL_DISPLAY.get(str(value), str(value)))


def display_best_model(best_model: str) -> str:
    return MODEL_DISPLAY.get(best_model, best_model)


def latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def fmt_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def bold_if(value: str, is_best: bool, latex: bool) -> str:
    if not is_best:
        return value
    return rf"\textbf{{{value}}}" if latex else f"**{value}**"


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print_saved(path)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def table1_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_key"] = df["model"].map(model_key)
    df["model_display"] = df["model"].map(display_model)
    df["is_baseline"] = df["model_key"] == "resnet50"
    baseline = df[df["is_baseline"]]
    others = df[~df["is_baseline"]].sort_values("f1_macro", ascending=True)
    return pd.concat([baseline, others], ignore_index=True)


def export_table1() -> pd.DataFrame | None:
    df = read_csv_checked(TEST_RESULTS_CSV, TEST_REQUIRED)
    if df is None:
        return None

    df = table1_sort(df)
    best_rules = {
        "accuracy": "max",
        "precision": "max",
        "recall": "max",
        "f1_macro": "max",
        "params_M": "min",
        "inference_ms": "min",
        "vram_peak_GB": "min",
    }
    best_values = {
        column: (df[column].max() if rule == "max" else df[column].min())
        for column, rule in best_rules.items()
    }

    latex_rows: list[str] = []
    md_rows: list[list[str]] = []
    for _, row in df.iterrows():
        formatted = {
            "Model": row["model_display"],
            "Accuracy (%)": fmt_percent(row["accuracy"]),
            "Precision (%)": fmt_percent(row["precision"]),
            "Recall (%)": fmt_percent(row["recall"]),
            "F1-macro (%)": fmt_percent(row["f1_macro"]),
            "Params (M)": fmt_float(row["params_M"], 1),
            "Infer (ms/img)": fmt_float(row["inference_ms"], 2),
            "VRAM (GB)": fmt_float(row["vram_peak_GB"], 2),
        }

        latex_cells = [
            latex_escape(formatted["Model"]),
            bold_if(formatted["Accuracy (%)"], row["accuracy"] == best_values["accuracy"], latex=True),
            bold_if(formatted["Precision (%)"], row["precision"] == best_values["precision"], latex=True),
            bold_if(formatted["Recall (%)"], row["recall"] == best_values["recall"], latex=True),
            bold_if(formatted["F1-macro (%)"], row["f1_macro"] == best_values["f1_macro"], latex=True),
            bold_if(formatted["Params (M)"], row["params_M"] == best_values["params_M"], latex=True),
            bold_if(formatted["Infer (ms/img)"], row["inference_ms"] == best_values["inference_ms"], latex=True),
            bold_if(formatted["VRAM (GB)"], row["vram_peak_GB"] == best_values["vram_peak_GB"], latex=True),
        ]
        md_cells = [
            formatted["Model"],
            bold_if(formatted["Accuracy (%)"], row["accuracy"] == best_values["accuracy"], latex=False),
            bold_if(formatted["Precision (%)"], row["precision"] == best_values["precision"], latex=False),
            bold_if(formatted["Recall (%)"], row["recall"] == best_values["recall"], latex=False),
            bold_if(formatted["F1-macro (%)"], row["f1_macro"] == best_values["f1_macro"], latex=False),
            bold_if(formatted["Params (M)"], row["params_M"] == best_values["params_M"], latex=False),
            bold_if(formatted["Infer (ms/img)"], row["inference_ms"] == best_values["inference_ms"], latex=False),
            bold_if(formatted["VRAM (GB)"], row["vram_peak_GB"] == best_values["vram_peak_GB"], latex=False),
        ]
        latex_rows.append(" & ".join(latex_cells) + r" \\")
        md_rows.append(md_cells)

    latex = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Performance comparison of deep learning models on the Eucalyptus wood species test set. Best results in each column are \textbf{bold}.}",
            r"\label{tab:benchmark}",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"Model & Accuracy (\%) & Precision (\%) & Recall (\%) & F1-macro (\%) & Params (M) & Infer (ms/img) & VRAM (GB) \\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    write_text(TABLES_DIR / "table1_benchmark.tex", latex)

    headers = [
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1-macro (%)",
        "Params (M)",
        "Infer (ms/img)",
        "VRAM (GB)",
    ]
    write_text(TABLES_DIR / "table1_benchmark.md", markdown_table(headers, md_rows))
    return df


def species_latex(value: str) -> str:
    return SPECIES_LATEX.get(value, latex_escape(value.replace("_", " ")))


def species_md(value: str) -> str:
    return SPECIES_MD.get(value, value.replace("_", " "))


def export_table2(best_model: str) -> pd.DataFrame | None:
    df = read_csv_checked(PER_CLASS_CSV, PER_CLASS_REQUIRED)
    if df is None:
        return None

    df = df.copy().sort_values("f1", ascending=True).reset_index(drop=True)
    lowest_idx = df["f1"].idxmin()
    highest_idx = df["f1"].idxmax()

    latex_rows: list[str] = []
    md_rows: list[list[str]] = []
    for idx, row in df.iterrows():
        prefix = ""
        if idx == lowest_idx:
            prefix = r"\rowcolor{red!15} "
        elif idx == highest_idx:
            prefix = r"\rowcolor{green!15} "

        latex_cells = [
            species_latex(row["class"]),
            fmt_percent(row["precision"]),
            fmt_percent(row["recall"]),
            fmt_percent(row["f1"]),
            str(int(row["support"])),
        ]
        md_cells = [
            species_md(row["class"]),
            fmt_percent(row["precision"]),
            fmt_percent(row["recall"]),
            fmt_percent(row["f1"]),
            str(int(row["support"])),
        ]
        latex_rows.append(prefix + " & ".join(latex_cells) + r" \\")
        md_rows.append(md_cells)

    macro_precision = df["precision"].mean()
    macro_recall = df["recall"].mean()
    macro_f1 = df["f1"].mean()
    support_sum = int(df["support"].sum())
    latex_rows.append(r"\midrule")
    latex_rows.append(
        r"\textbf{Macro avg} & "
        + rf"\textbf{{{fmt_percent(macro_precision)}}} & "
        + rf"\textbf{{{fmt_percent(macro_recall)}}} & "
        + rf"\textbf{{{fmt_percent(macro_f1)}}} & "
        + rf"\textbf{{{support_sum}}} \\"
    )
    md_rows.append(
        [
            "**Macro avg**",
            f"**{fmt_percent(macro_precision)}**",
            f"**{fmt_percent(macro_recall)}**",
            f"**{fmt_percent(macro_f1)}**",
            f"**{support_sum}**",
        ]
    )

    latex = "\n".join(
        [
            r"% Requires \usepackage{booktabs} and \usepackage{colortbl}",
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{Per-class classification metrics of {latex_escape(display_best_model(best_model))} on the test set. Species sorted by F1-score (ascending).}}",
            r"\label{tab:per_class}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Species & Precision (\%) & Recall (\%) & F1 (\%) & $n$ \\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    write_text(TABLES_DIR / "table2_per_class.tex", latex)

    headers = ["Species", "Precision (%)", "Recall (%)", "F1 (%)", "n"]
    write_text(TABLES_DIR / "table2_per_class.md", markdown_table(headers, md_rows))
    return df


def load_ablation_inputs() -> dict[str, pd.DataFrame] | None:
    inputs = {
        "resolution": ABLATION_RESOLUTION_CSV,
        "augmentation": ABLATION_AUGMENTATION_CSV,
        "finetuning": ABLATION_FINETUNING_CSV,
    }
    loaded: dict[str, pd.DataFrame] = {}
    missing_any = False
    for group, path in inputs.items():
        df = read_csv_checked(path, ABLATION_REQUIRED)
        if df is None:
            missing_any = True
        else:
            loaded[group] = df
    if missing_any:
        return None
    return loaded


def ablation_setting_display(group: str, setting: str, latex: bool) -> str:
    normalized = str(setting).strip().lower()
    if group == "resolution":
        display = "224×224" if normalized in {"224x224", "224×224"} else "384×384" if normalized in {"384x384", "384×384"} else str(setting)
        if display == "224×224":
            display += "*"
        return display

    if group == "augmentation":
        if normalized == "none":
            return "None"
        if normalized == "basic":
            return "Basic"
        if normalized == "full":
            return "Full augmentation*"
        return str(setting)

    if group == "finetuning":
        if normalized == "frozen":
            return "Frozen"
        if normalized == "partial":
            return "Partial"
        if normalized == "full":
            return "Full fine-tune*"
        return str(setting)

    return str(setting)


def section_rows(group: str, df: pd.DataFrame, latex: bool) -> list[list[str]]:
    best_test_f1 = df["test_f1"].max()
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        setting = ablation_setting_display(group, row["setting"], latex=latex)
        test_f1 = fmt_percent(row["test_f1"])
        rows.append(
            [
                setting,
                fmt_percent(row["val_f1"]),
                fmt_percent(row["test_acc"]),
                bold_if(test_f1, row["test_f1"] == best_test_f1, latex=latex),
                fmt_float(row["training_time_min"], 1),
            ]
        )
    return rows


def export_table3(best_model: str) -> None:
    loaded = load_ablation_inputs()
    if loaded is None:
        return

    section_specs = [
        ("resolution", "(a) Image Resolution"),
        ("augmentation", "(b) Data Augmentation"),
        ("finetuning", "(c) Fine-tuning Strategy"),
    ]

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Ablation study results using {latex_escape(display_best_model(best_model))} (20 epochs each, seed=42).}}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Setting & Val F1 (\%) & Test Acc (\%) & Test F1 (\%) & Time (min) \\",
        r"\midrule",
    ]
    md_lines: list[str] = []

    for section_idx, (group, title) in enumerate(section_specs):
        if section_idx > 0:
            latex_lines.append(r"\midrule")
            md_lines.append("")
        latex_lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\")
        rows = section_rows(group, loaded[group], latex=True)
        for row in rows:
            latex_lines.append(" & ".join(row) + r" \\")

        md_lines.append(f"### {title}")
        md_lines.append(markdown_table(["Setting", "Val F1 (%)", "Test Acc (%)", "Test F1 (%)", "Time (min)"], section_rows(group, loaded[group], latex=False)).rstrip())

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4em}",
            r"\begin{flushleft}",
            r"\footnotesize{* Baseline configuration used in Table 1.}",
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    write_text(TABLES_DIR / "table3_ablation.tex", "\n".join(latex_lines))

    md_lines.append("")
    md_lines.append("* Baseline configuration used in Table 1.")
    write_text(TABLES_DIR / "table3_ablation.md", "\n".join(md_lines) + "\n")


def print_abstract_stats(benchmark_df: pd.DataFrame | None, per_class_df: pd.DataFrame | None) -> None:
    print("\nAbstract-ready stats:", flush=True)

    if benchmark_df is not None and not benchmark_df.empty:
        best_row = benchmark_df.loc[benchmark_df["f1_macro"].idxmax()]
        baseline_df = benchmark_df[benchmark_df["model_key"] == "resnet50"]

        print(
            f"Best model: {best_row['model_display']} — Test Accuracy: {best_row['accuracy'] * 100:.2f}%, "
            f"F1-macro: {best_row['f1_macro'] * 100:.2f}%",
            flush=True,
        )
        if not baseline_df.empty:
            baseline = baseline_df.iloc[0]
            delta = (best_row["f1_macro"] - baseline["f1_macro"]) * 100
            print(
                f"Baseline (ResNet-50): Accuracy {baseline['accuracy'] * 100:.2f}%, "
                f"F1 {baseline['f1_macro'] * 100:.2f}%",
                flush=True,
            )
            print(f"Improvement over baseline: +{delta:.2f}% F1", flush=True)
        else:
            print("Baseline (ResNet-50): unavailable", flush=True)
            print("Improvement over baseline: unavailable", flush=True)
    else:
        print("Best model: unavailable", flush=True)
        print("Baseline (ResNet-50): unavailable", flush=True)
        print("Improvement over baseline: unavailable", flush=True)

    if per_class_df is not None and not per_class_df.empty:
        lowest = per_class_df.loc[per_class_df["f1"].idxmin()]
        highest = per_class_df.loc[per_class_df["f1"].idxmax()]
        print(f"Lowest F1 class: {species_md(lowest['class'])} ({lowest['f1'] * 100:.2f}%)", flush=True)
        print(f"Highest F1 class: {species_md(highest['class'])} ({highest['f1'] * 100:.2f}%)", flush=True)
    else:
        print("Lowest F1 class: unavailable", flush=True)
        print("Highest F1 class: unavailable", flush=True)

    if SPLIT_STATS_CSV.exists():
        split_df = pd.read_csv(SPLIT_STATS_CSV)
        required = {"n_train", "n_val", "n_test"}
        if required.issubset(split_df.columns):
            n_train = int(split_df["n_train"].sum())
            n_val = int(split_df["n_val"].sum())
            n_test = int(split_df["n_test"].sum())
            print(f"Dataset: 8 species, {n_train} train / {n_val} val / {n_test} test images", flush=True)
        else:
            print(f"✗ Missing columns in {SPLIT_STATS_CSV.name}: {sorted(required - set(split_df.columns))}", flush=True)
    else:
        print_missing(SPLIT_STATS_CSV)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaTeX and Markdown paper tables from result CSVs.")
    parser.add_argument("--best_model", default=BEST_MODEL, choices=sorted(set(MODEL_KEY_ALIASES.values())), help="Best model key for captions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_df = export_table1()
    per_class_df = export_table2(args.best_model)
    export_table3(args.best_model)
    print_abstract_stats(benchmark_df, per_class_df)


if __name__ == "__main__":
    main()


# python export_paper_tables.py
# python export_paper_tables.py --best_model convnext_tiny
