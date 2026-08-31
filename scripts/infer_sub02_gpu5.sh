#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-../neuro-3D-main_V100/EEG_3Datasets}"
RESULT_DIR="${RESULT_DIR:-./neuro3d_result}"
OUTPUT_DIR="${OUTPUT_DIR:-./PointCNN/outputs}"
SUBJECT="${SUBJECT:-sub02}"
GPU_ID="${GPU_ID:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-double_10w_2048_dp8_EnhanceGD_only_mse}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RESULT_DIR/$SUBJECT/$EXPERIMENT_NAME/checkpoint-100000.pth}"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}" \
python -u PointCNN/Test.py \
  --device cuda \
  --sub "$SUBJECT" \
  --data_path "$DATA_PATH"/ \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --infer_output_dir "$OUTPUT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --in_channels 1027 \
  --test_batch_size 4 \
  --num_points 2048 \
  --infer_repeat 5

echo "Inference finished for $SUBJECT"
echo "Output: $OUTPUT_DIR/$SUBJECT/$EXPERIMENT_NAME"
