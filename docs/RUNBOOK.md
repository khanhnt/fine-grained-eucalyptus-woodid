# Reproduction Runbook

## 1. Materialize Split B

```bash
python scripts/materialize_split.py \
  --raw-root /path/to/IC4SD-Wood-Eucalyptus/raw \
  --split-csv manifests/split_B_strict.csv \
  --output-dir data \
  --results-dir results \
  --copy
```

This creates:

- `data/train`, `data/val`, `data/test`
- `results/label_map.json`
- `results/current_split_split_B_strict.csv`

## 2. Train All Models

```bash
nohup bash run_splitB_4models_3seeds.sh > results/splitB_4models_3seeds_master.log 2>&1 &
```

## 3. Resume or Rerun One Model/Seed

```bash
python train_benchmark.py --model efficientnet_b4 --epochs 50 --batch_size 64 --seed 3407
python train_benchmark.py --model resnet50 --epochs 50 --batch_size 64 --seed 3407
python train_benchmark.py --model vit_b16 --epochs 50 --vit_b16_batch_size 24 --seed 3407
```

## 4. Generate Figures

```bash
ln -sf ConvNeXtTiny_seed3407_best.pth checkpoints/convnext_tiny_best.pth
python analyze_results.py --best_model convnext_tiny --batch_size 128
python gradcam_visualization.py --best_model convnext_tiny --n_samples 3 --batch_size 128
```

## 5. Leakage Audit

```bash
python scripts/run_leakage_audit.py \
  --data-root data \
  --split-file results/current_split_split_B_strict.csv \
  --out-dir results/leakage_audit_splitB \
  --phash-thresholds 5 10 \
  --top-k 5
```

## 6. Calibration and OOD

```bash
python scripts/evaluate_calibration_ood.py \
  --checkpoint checkpoints/ConvNeXtTiny_seed3407_best.pth \
  --known-split-file results/current_split_split_B_strict.csv \
  --ood-data-root /path/to/Eucalyptus_globulus_external \
  --model convnext_tiny \
  --img-size 224 \
  --out-dir results/calibration_ood_splitB_convnext_seed3407
```
