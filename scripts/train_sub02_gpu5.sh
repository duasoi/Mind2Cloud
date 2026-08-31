#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-../neuro-3D-main_V100/EEG_3Datasets}"
RESULT_DIR="${RESULT_DIR:-./neuro3d_result}"
LOG_DIR="${LOG_DIR:-./logs}"
SUBJECT="${SUBJECT:-sub02}"
GPU_ID="${GPU_ID:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-double_10w_2048_dp8_EnhanceGD_only_mse}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-100000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5000}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
nohup python -u PointCNN/Train.py \
  --device cuda \
  --task train \
  --sub "$SUBJECT" \
  --data_path "$DATA_PATH"/ \
  --output_dir "$RESULT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --in_channels 1027 \
  --max_steps "$MAX_STEPS" \
  --checkpoint_freq "$CHECKPOINT_FREQ" \
  --log_step_freq 10 \
  --batch_size "$BATCH_SIZE" \
  --val_batch_size "$VAL_BATCH_SIZE" \
  --num_workers 4 \
  > "$LOG_DIR/${SUBJECT}_shape_gpu${GPU_ID}.log" 2>&1 &

echo "Started $SUBJECT on GPU $GPU_ID"
echo "Log: $LOG_DIR/${SUBJECT}_shape_gpu${GPU_ID}.log"
