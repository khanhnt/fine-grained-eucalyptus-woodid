# Fine-Grained Eucalyptus WoodID

Companion research code for the manuscript:

**Deep Learning for Fine-Grained Eucalyptus Species Identification Using Macroscopic Wood Cross-Section Images**

This repository contains the code, split manifests, and summary files needed to reproduce the main machine-learning experiments reported in the manuscript. It is separate from the dataset release repository so that the research-code package remains focused on model training, evaluation, robustness analysis, and figure/table generation.

## Description

The project evaluates deep learning models for fine-grained classification of macroscopic transverse-section wood images from eight *Eucalyptus*/*Syzygium* species. The main experiments compare four ImageNet-pretrained architectures:

- ResNet-50
- EfficientNet-B4
- ConvNeXt-Tiny
- ViT-B/16

The repository supports:

- Split A and Split B benchmark reproduction.
- Four-model training and evaluation across three random seeds.
- Leakage auditing with group identifiers, exact file hashes, filenames, perceptual hashes, and feature-nearest-neighbor checks.
- Calibration and unseen-species/OOD diagnostics using an external *Eucalyptus globulus* image set.
- Confusion matrix, learning curves, t-SNE, Grad-CAM, and LaTeX-ready table generation.

Raw images and trained checkpoints are not stored in this repository because of size and data-release management. The included scripts materialize the required `train/val/test` folder structure from published split manifests.

## Dataset Information

Dataset name:

```text
IC4SD-Wood-Eucalyptus
```

Image type:

```text
Macroscopic transverse-section wood images
```

Magnification:

```text
50x
```

Number of classes:

```text
8
```

Classes:

```text
Eucalyptus camaldulensis
Eucalyptus cladocalyx
Eucalyptus deglupta
Eucalyptus diversicolor
Eucalyptus grandis
Eucalyptus microcorys
Eucalyptus saligna
Syzygium hemisphericum
```

The dataset is expected to be available separately as a raw image release. The raw dataset folder should contain one top-level folder per species, with images possibly stored directly inside the species folder or in nested acquisition/specimen folders.

This repository includes split manifests under `manifests/`:

```text
manifests/split_A_reference.csv
manifests/split_B_strict.csv
manifests/label_map.json
manifests/dataset_release_summary.json
manifests/release_counts.csv
```

Split B is the main strict split used for the final robustness experiments. It contains:

```text
train: 2029 images
val:    486 images
test:   395 images
total: 2910 images
```

## Code Information

Main files:

```text
train_benchmark.py                 Train/evaluate ResNet-50, EfficientNet-B4, ConvNeXt-Tiny, ViT-B/16
analyze_results.py                 Confusion matrix, per-class F1, learning curves, t-SNE
gradcam_visualization.py           Grad-CAM visualizations
export_paper_tables.py             LaTeX/Markdown table export
run_splitB_4models_3seeds.sh       Full Split B training launcher
```

Utility scripts:

```text
scripts/materialize_split.py        Build ImageFolder train/val/test folders from a split CSV
scripts/run_leakage_audit.py        Leakage and near-duplicate audit
scripts/evaluate_calibration_ood.py Calibration and OOD/unseen-species evaluation
scripts/run_repeated_splits.py      Repeated group-disjoint split experiments
scripts/paper_utils.py              Shared model, data, metric, and plotting utilities
```

Summary and supplementary result files:

```text
results_summary/test_results_splitB_4models_3seeds.csv
results_summary/splitB_final_multiseed_summary.csv
supplementary/training_logs/
manifests/leakage_audit_reports/
```

## Requirements

Recommended environment:

```text
Python >= 3.10
CUDA-capable NVIDIA GPU for training
PyTorch 2.x
torchvision
timm
scikit-learn
pandas
numpy
matplotlib
Pillow
tqdm
grad-cam
opencv-python-headless
einops
imagehash
```

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA servers, install a PyTorch build compatible with the installed NVIDIA driver. For example, on CUDA 12.4 systems:

```bash
python -m pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Usage Instructions

### 1. Prepare the raw dataset

Place or mount the raw dataset in a folder such as:

```text
/path/to/IC4SD-Wood-Eucalyptus/raw
```

Expected structure:

```text
raw/
  Eucalyptus_camaldulensis/
    image_or_subfolder_files...
  Eucalyptus_cladocalyx/
    image_or_subfolder_files...
  Eucalyptus_deglupta/
    image_or_subfolder_files...
  Eucalyptus_diversicolor/
    image_or_subfolder_files...
  Eucalyptus_grandis/
    image_or_subfolder_files...
  Eucalyptus_microcorys/
    image_or_subfolder_files...
  Eucalyptus_saligna/
    image_or_subfolder_files...
  Syzygium_hemisphericum/
    image_or_subfolder_files...
```

### 2. Materialize Split B

The training scripts use an ImageFolder-style directory. Create it from the Split B manifest:

```bash
python scripts/materialize_split.py \
  --raw-root /path/to/IC4SD-Wood-Eucalyptus/raw \
  --split-csv manifests/split_B_strict.csv \
  --output-dir data \
  --results-dir results \
  --copy
```

This creates:

```text
data/train/
data/val/
data/test/
results/label_map.json
results/current_split_split_B_strict.csv
```

The class names in the split manifest use scientific display names such as `Eucalyptus camaldulensis`; the materialization script converts them to ImageFolder-safe names such as `Eucalyptus_camaldulensis`.

### 3. Train all four models across three seeds

Run the full Split B benchmark:

```bash
nohup bash run_splitB_4models_3seeds.sh > results/splitB_4models_3seeds_master.log 2>&1 &
```

Monitor progress:

```bash
tail -f results/final_splitB_stdout/convnext_tiny_seed3407.log
watch -n5 nvidia-smi
```

### 4. Train a single model

Examples:

```bash
python train_benchmark.py --model convnext_tiny --epochs 50 --batch_size 64 --seed 3407
python train_benchmark.py --model resnet50 --epochs 50 --batch_size 64 --seed 42
python train_benchmark.py --model efficientnet_b4 --epochs 50 --batch_size 64 --seed 2025
python train_benchmark.py --model vit_b16 --epochs 50 --vit_b16_batch_size 24 --seed 3407
```

### 5. Generate analysis figures

After training, make sure the best checkpoint and test split are available:

```bash
ln -sf ConvNeXtTiny_seed3407_best.pth checkpoints/convnext_tiny_best.pth
python analyze_results.py --best_model convnext_tiny --batch_size 128
```

Expected figure outputs:

```text
results/figures/confusion_matrix_8x8.png
results/figures/per_class_f1.png
results/figures/learning_curves_all_models.png
results/figures/tsne_features.png
```

### 6. Generate Grad-CAM figures

```bash
python gradcam_visualization.py \
  --best_model convnext_tiny \
  --n_samples 3 \
  --batch_size 128
```

Expected outputs:

```text
results/figures/gradcam_main_grid.png
results/figures/gradcam_misclassified.png
results/figures/gradcam_confused_pairs.png
results/gradcam_individual/
```

### 7. Run leakage audit

```bash
python scripts/run_leakage_audit.py \
  --data-root data \
  --split-file results/current_split_split_B_strict.csv \
  --out-dir results/leakage_audit_splitB \
  --phash-thresholds 5 10 \
  --top-k 5
```

Expected outputs include:

```text
results/leakage_audit_splitB/leakage_audit_report.md
results/leakage_audit_splitB/leakage_group_audit.csv
results/leakage_audit_splitB/leakage_exact_hash_audit.csv
results/leakage_audit_splitB/leakage_filename_audit.csv
results/leakage_audit_splitB/leakage_phash_pairs.csv
results/leakage_audit_splitB/leakage_feature_nn_pairs.csv
results/leakage_audit_splitB/contact_sheets/
```

### 8. Run calibration and OOD evaluation

Use the best ConvNeXt-Tiny checkpoint and an external unseen-species folder, for example external *Eucalyptus globulus* images:

```bash
python scripts/evaluate_calibration_ood.py \
  --checkpoint checkpoints/ConvNeXtTiny_seed3407_best.pth \
  --known-split-file results/current_split_split_B_strict.csv \
  --ood-data-root /path/to/Eucalyptus_globulus_external \
  --model convnext_tiny \
  --img-size 224 \
  --out-dir results/calibration_ood_splitB_convnext_seed3407
```

Expected outputs include:

```text
known_confidence_scores.csv
ood_confidence_scores.csv
calibration_metrics.csv
ood_metrics.csv
ood_threshold_analysis.csv
ood_forced_prediction_distribution.csv
reliability_diagram.png
confidence_histogram_known_vs_ood.png
entropy_histogram_known_vs_ood.png
energy_histogram_known_vs_ood.png
ood_roc_curve.png
ood_pr_curve.png
```

## Methodology

### Split protocol

The experiments use specimen/acquisition-aware split manifests. Split B is the main strict split used for robustness claims. It was designed to reduce near-duplicate leakage risk using group-disjoint splitting and perceptual-hash screening.

### Training

The main benchmark uses:

```text
Epochs: 50
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-2
Scheduler: CosineAnnealingLR
Loss: CrossEntropyLoss with label_smoothing=0.1
Input size: 224 for ResNet-50, EfficientNet-B4, ConvNeXt-Tiny
Input size: 384 for ViT-B/16
Seeds: 42, 2025, 3407
```

Training augmentation:

```text
RandomHorizontalFlip
RandomVerticalFlip
RandomRotation(30)
ColorJitter
ImageNet normalization
```

Validation and test preprocessing:

```text
Resize
CenterCrop
ImageNet normalization
```

### Evaluation

The scripts compute:

```text
Accuracy
Macro precision
Macro recall
Macro F1
Weighted F1
Per-class precision/recall/F1
Confusion matrix
Inference latency
GPU memory usage
```

Calibration/OOD analysis computes:

```text
NLL
Brier score
ECE
MCE
Maximum softmax probability
Entropy
Energy score
AUROC
AUPR-In
AUPR-Out
FPR@95TPR
Threshold-based known/OOD rejection statistics
```

## Reproducibility Notes

- Set all random seeds through the script CLI.
- Keep the exact split manifest used for each experiment.
- Do not mix Split A and Split B results in the same output directory.
- Use the same raw image release and split manifest when reproducing manuscript tables.
- Check `results/logs/` for per-epoch learning-curve CSV files generated by a new run.
- Check `supplementary/training_logs/` for the archived per-model/per-seed learning-curve CSV files used for the manuscript.
- Check `results_summary/` for manuscript-ready aggregate CSV files.
- Check `manifests/leakage_audit_reports/` for the Split A vs Split B pHash audit CSV files.

## Citations

If you use this repository, cite the associated dataset and research manuscript:

```text
Deep Learning for Fine-Grained Eucalyptus Species Identification Using
Macroscopic Wood Cross-Section Images.
```

Also cite the dataset release:

```text
IC4SD-Wood-Eucalyptus dataset release.
```

The final DOI, journal name, volume, pages, and year should be cited when the article is formally published. At the review stage, cite the manuscript title and dataset release name shown above.

## License and Contribution Guidelines

This repository is intended for academic reproducibility and manuscript review.

- Raw images are not distributed in this repository.
- Dataset use should follow the license and access conditions of the dataset release.
- No standalone software license is currently declared in this repository.
- Until a formal `LICENSE` file is added, the code is provided for academic review and reproducibility checking.
- Contributions should preserve reproducibility: include configuration files, random seeds, split manifests, and clear output paths for any new experiment.

## Contact

For questions about the dataset, split manifests, or reproduction of the manuscript experiments, contact the corresponding author listed in the manuscript.
