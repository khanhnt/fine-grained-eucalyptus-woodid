# Fine-Grained Eucalyptus WoodID

Companion code for the manuscript:

**Deep Learning for Fine-Grained Eucalyptus Species Identification Using Macroscopic Wood Cross-Section Images**

This repository is intentionally separate from the Data in Brief dataset repository. It contains the research-article experiment code used for:

- Split A / Split B benchmark reproduction.
- Four-model benchmark: ResNet-50, EfficientNet-B4, ConvNeXt-Tiny, and ViT-B/16.
- Three-seed robustness evaluation on Split B.
- Leakage auditing with group, hash, filename, pHash, and feature-nearest-neighbor checks.
- Calibration and OOD/unseen-species diagnostics.
- Confusion matrix, learning curves, t-SNE, Grad-CAM, and LaTeX-ready summary tables.

The repository does **not** include raw images, checkpoints, or long-running generated outputs. Use the public dataset release and place/materialize data locally as described below.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── train_benchmark.py
├── analyze_results.py
├── gradcam_visualization.py
├── export_paper_tables.py
├── run_splitB_4models_3seeds.sh
├── configs/
├── docs/
├── manifests/
│   ├── split_A_reference.csv
│   ├── split_B_strict.csv
│   ├── label_map.json
│   ├── dataset_release_summary.json
│   └── release_counts.csv
├── results_summary/
└── scripts/
    ├── materialize_split.py
    ├── paper_utils.py
    ├── run_leakage_audit.py
    ├── evaluate_calibration_ood.py
    └── run_repeated_splits.py
```

## Data Assumption

The benchmark scripts expect an ImageFolder-style local directory:

```text
data/
  train/
    Eucalyptus_camaldulensis/
    Eucalyptus_cladocalyx/
    ...
  val/
    ...
  test/
    ...
```

The `manifests/split_B_strict.csv` file uses scientific display names such as `Eucalyptus camaldulensis`; `scripts/materialize_split.py` converts these to ImageFolder-safe class folders such as `Eucalyptus_camaldulensis`.

## Quick Start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Materialize Split B from the public raw dataset:

```bash
python scripts/materialize_split.py \
  --raw-root /path/to/IC4SD-Wood-Eucalyptus/raw \
  --split-csv manifests/split_B_strict.csv \
  --output-dir data \
  --results-dir results \
  --copy
```

Run the four-model, three-seed Split B benchmark:

```bash
nohup bash run_splitB_4models_3seeds.sh > results/splitB_4models_3seeds_master.log 2>&1 &
```

Monitor:

```bash
tail -f results/final_splitB_stdout/convnext_tiny_seed3407.log
watch -n5 nvidia-smi
```

## Single-Model Examples

```bash
python train_benchmark.py --model convnext_tiny --epochs 50 --batch_size 64 --seed 3407
python train_benchmark.py --model resnet50 --epochs 50 --batch_size 64 --seed 42
python train_benchmark.py --model vit_b16 --epochs 50 --vit_b16_batch_size 24 --seed 2025
```

## Analysis Figures

After training, make sure `results/label_map.json`, `data/test/`, and the best ConvNeXt-Tiny checkpoint exist.

```bash
ln -sf ConvNeXtTiny_seed3407_best.pth checkpoints/convnext_tiny_best.pth
python analyze_results.py --best_model convnext_tiny --batch_size 128
python gradcam_visualization.py --best_model convnext_tiny --n_samples 3 --batch_size 128
```

Expected outputs include:

- `results/figures/confusion_matrix_8x8.png`
- `results/figures/per_class_f1.png`
- `results/figures/learning_curves_all_models.png`
- `results/figures/tsne_features.png`
- `results/figures/gradcam_main_grid.png`
- `results/figures/gradcam_misclassified.png`

## Leakage Audit

For an already materialized ImageFolder split:

```bash
python scripts/run_leakage_audit.py \
  --data-root data \
  --split-file results/current_split_splitB.csv \
  --out-dir results/leakage_audit_splitB \
  --phash-thresholds 5 10 \
  --top-k 5
```

For manuscript text, describe the result carefully, for example:

> No group-overlap, exact-hash, or filename-duplicate evidence was found under the performed audits. pHash and feature-nearest-neighbor analyses were used as additional near-duplicate screening tools.

## Calibration and OOD

Use the best ConvNeXt-Tiny checkpoint and the known test split. The OOD folder should contain external unseen-species images, e.g. *Eucalyptus globulus*.

```bash
python scripts/evaluate_calibration_ood.py \
  --checkpoint checkpoints/ConvNeXtTiny_seed3407_best.pth \
  --known-split-file results/current_split_splitB.csv \
  --ood-data-root /path/to/Eucalyptus_globulus_external \
  --model convnext_tiny \
  --img-size 224 \
  --out-dir results/calibration_ood_splitB_convnext_seed3407
```

## Final Split B Results

The manuscript-ready final CSVs are in `results_summary/`:

- `test_results_splitB_4models_3seeds.csv`
- `splitB_final_multiseed_summary.csv`

These contain the cleaned final Split B results after fresh reruns of the affected seed-3407 ResNet-50 and ViT-B/16 runs.

## Notes

- Do not commit raw image data or checkpoints to this repository.
- Keep Split A as a reference/baseline split and Split B as the strict pHash-clean split used for the main robustness claims.
- The OOD/unseen-species experiment should not be described as closed-set accuracy. It characterizes how a closed-set classifier behaves when exposed to an unseen species.
