#!/bin/bash

# Usage: ./train_100_dr.sh [gpu_ids] [curriculum] [threshold]
# Trains with DR, soft pseudo labels and curriculum weight
# 
# Arguments:
#   gpu_ids: Comma-separated GPU IDs (default: "0,1")
#   curriculum: Curriculum type - none, linear, or quadratic (default: "quadratic")
#   threshold: Threshold for pseudo-label masking (default: "0.5")

GPU_IDS=${1:-"0,1"}
CURRICULUM=${2:-"quadratic"}
THRESHOLD=${3:-"0."}

# Calculate num_processes from GPU IDs (count commas + 1)
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Generate a random port for distributed training (29500-30000 range)
RANDOM_PORT=$((29500 + RANDOM % 500))

echo "Using GPU IDs: $GPU_IDS (num_processes=$NUM_PROCESSES)"
echo "Using curriculum: $CURRICULUM"
echo "Using threshold: $THRESHOLD"
echo "Using random port: $RANDOM_PORT"

# Build output directory name based on arguments
TRAIN_OUTPUT_BASE_DIR="train_outputs_dr_v1"
TEST_OUTPUT_BASE_DIR="test_outputs_dr_v1"
EXP_NAME="sd15_25_75_steps=100_dr_curriculum=${CURRICULUM}_threshold=${THRESHOLD}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_BASE_DIR}/${EXP_NAME}"

echo "Running: train 25/75 split 5k datapoints with DR, soft pseudo labels, $CURRICULUM curriculum weight, threshold=$THRESHOLD (DRST)"

accelerate launch --gpu_ids "$GPU_IDS" --num_processes=$NUM_PROCESSES --main_process_port=$RANDOM_PORT train_sd15_dpo_dr.py \
    --mixed_precision "fp16" \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --labeled_manifest "datasets/manifest/25_75/labeled.json" \
    --unlabeled_manifest "datasets/manifest/25_75/unlabeled.json" \
    --train_data_dir "datasets/pickapic_v2/" \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 100 \
    --lr_warmup_steps 10 \
    --learning_rate 1e-7 \
    --scale_lr \
    --checkpointing_steps 10000 \
    --beta_dpo 5000 \
    --report_to "wandb" \
    --dataloader_num_workers 4 \
    --mu 3 \
    --curriculum "$CURRICULUM" \
    --threshold "$THRESHOLD"

echo "Running: test 25/75 split 5k datapoints with DR, soft pseudo labels, $CURRICULUM curriculum weight, threshold=$THRESHOLD (DRST)"
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
