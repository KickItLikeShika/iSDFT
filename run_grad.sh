#!/usr/bin/env bash
set -euo pipefail

# Keep the existing training and vLLM settings; only enable the sequence-score correction.
OUTPUT_DIR=${1:-outputs/tooluse-sequence-score}

python main.py \
  --dataset_name tooluse \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir "${OUTPUT_DIR}" \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --gradient_estimator sequence_score
