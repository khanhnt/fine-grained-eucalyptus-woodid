#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEEDS=(42 2025 3407)

mkdir -p results/final_splitB_stdout results/logs checkpoints

for seed in "${SEEDS[@]}"; do
  echo "===== Seed ${seed} | ResNet50 ====="
  python train_benchmark.py --model resnet50 --epochs 50 --batch_size 64 --seed "$seed" \
    > "results/final_splitB_stdout/resnet50_seed${seed}.log" 2>&1

  echo "===== Seed ${seed} | EfficientNet-B4 ====="
  python train_benchmark.py --model efficientnet_b4 --epochs 50 --batch_size 64 --seed "$seed" \
    > "results/final_splitB_stdout/efficientnet_b4_seed${seed}.log" 2>&1

  echo "===== Seed ${seed} | ConvNeXt-Tiny ====="
  python train_benchmark.py --model convnext_tiny --epochs 50 --batch_size 64 --seed "$seed" \
    > "results/final_splitB_stdout/convnext_tiny_seed${seed}.log" 2>&1

  echo "===== Seed ${seed} | ViT-B/16 ====="
  python train_benchmark.py --model vit_b16 --epochs 50 --vit_b16_batch_size 24 --seed "$seed" \
    > "results/final_splitB_stdout/vit_b16_seed${seed}.log" 2>&1
done

echo "DONE Split B 4 models x 3 seeds"
