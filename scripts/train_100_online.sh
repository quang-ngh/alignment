#!/bin/bash

# Usage: ./train_100_online.sh [gpu_ids] [mode] [threshold]
# Trains with online DPO and pseudo labels
# 
# Arguments:
#   gpu_ids: Comma-separated GPU IDs (default: "0,1")
#   mode: Training mode - "dr" or "pseudo" (default: "dr")
#   threshold: Threshold for pseudo-label masking (default: "0.")

GPU_IDS=${1:-"0,1"}
MODE=${2:-"dr"}
THRESHOLD=${3:-"0."}

# Calculate num_processes from GPU IDs (count commas + 1)
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Generate a random port for distributed training (29500-30000 range)
RANDOM_PORT=$((29500 + RANDOM % 500))

# Validate mode
if [ "$MODE" != "dr" ] && [ "$MODE" != "pseudo" ]; then
    echo "Error: Mode must be 'dr' or 'pseudo'"
    echo "Usage: ./train_100_online.sh [gpu_ids] [mode] [threshold]"
    exit 1
fi

echo "Using GPU IDs: $GPU_IDS (num_processes=$NUM_PROCESSES)"
echo "Using mode: $MODE"
echo "Using threshold: $THRESHOLD"
echo "Using random port: $RANDOM_PORT"

# Build output directory name based on arguments
TRAIN_OUTPUT_BASE_DIR="train_outputs_online_v1"
TEST_OUTPUT_BASE_DIR="test_outputs_online_v1"
EXP_NAME="sd15_5k_steps=100_${MODE}_threshold=${THRESHOLD}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_BASE_DIR}/${EXP_NAME}"

echo "Running: train 5k datapoints with online DPO and pseudo labels"

# Build command arguments
CMD_ARGS=(
    --mixed_precision "fp16"
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5"
    --output_dir "$TRAIN_OUTPUT_DIR"
    --manifest "datasets/manifest/fifa_5k_new.json"
    --train_data_dir "datasets/pickapic_v2/"
    --train_batch_size 4
    --gradient_accumulation_steps 8
    --max_train_steps 100
    --lr_warmup_steps 10
    --learning_rate 1e-7
    --scale_lr
    --checkpointing_steps 1000
    --beta_dpo 5000
    --report_to "wandb"
    --dataloader_num_workers 4
    --threshold "$THRESHOLD"
)

# Add --no_dr if mode is "pseudo"
if [ "$MODE" = "pseudo" ]; then
    CMD_ARGS+=(--no-dr)
fi

accelerate launch --gpu_ids "$GPU_IDS" --num_processes=$NUM_PROCESSES --main_process_port=$RANDOM_PORT train_sd15_dpo_online.py "${CMD_ARGS[@]}"

echo "Running: test 5k datapoints with online DPO and pseudo labels"
accelerate launch --gpu_ids "$GPU_IDS" --num_processes=$NUM_PROCESSES --main_process_port=$RANDOM_PORT fifa_test.py \
    --prompts_path "datasets/eval_prompts/pickapic_test.json" \
    --model-path "$TRAIN_OUTPUT_DIR" \
    --version "$EXP_NAME" \
    --dataset pickscore \
    --reward_type pickscore \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --output-dir "$TEST_OUTPUT_BASE_DIR" \
    --num_imgs_per_prompt 4 \
    --batch_size 32 \
    --num_inference_steps 20 \
    --overwrite 0