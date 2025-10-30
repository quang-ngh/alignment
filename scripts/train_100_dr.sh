#!/bin/bash

# Usage: ./train_100_dr.sh [gpu_ids] [curriculum] [hard_pseudo_label] [use_updated_policy]
# Trains with DR, soft pseudo labels and curriculum weight
# 
# Arguments:
#   gpu_ids: Comma-separated GPU IDs (default: "0,1")
#   curriculum: Curriculum type - none, linear, or quadratic (default: "quadratic")
#   hard_pseudo_label: Set to "true" to use hard pseudo labels, "false" for soft (default: "false")
#   use_updated_policy: Set to "true" to use updated policy for pseudo labels, "false" for reference policy (default: "false")

GPU_IDS=${1:-"0,1"}
CURRICULUM=${2:-"quadratic"}
HARD_PSEUDO_LABEL=${3:-"false"}
USE_UPDATED_POLICY=${4:-"false"}

# Calculate num_processes from GPU IDs (count commas + 1)
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "Using GPU IDs: $GPU_IDS (num_processes=$NUM_PROCESSES)"
echo "Using curriculum: $CURRICULUM"
echo "Using hard pseudo label: $HARD_PSEUDO_LABEL"
echo "Using updated policy for pseudo labels: $USE_UPDATED_POLICY"

# Determine pseudo label type for echo
if [ "$HARD_PSEUDO_LABEL" = "true" ]; then
    PSEUDO_TYPE="hard pseudo labels"
    PSEUDO_SUFFIX="_hard"
else
    PSEUDO_TYPE="soft pseudo labels"
    PSEUDO_SUFFIX="_soft"
fi

# Determine policy source for echo
if [ "$USE_UPDATED_POLICY" = "true" ]; then
    POLICY_SOURCE="updated"
    POLICY_SUFFIX="_updated"
else
    POLICY_SOURCE="reference"
    POLICY_SUFFIX="_ref"
fi

# Build output directory name based on arguments
OUTPUT_DIR="training_runs/sd15_25_75_steps=100_dr_${CURRICULUM}${PSEUDO_SUFFIX}${POLICY_SUFFIX}"
OUTPUT_VERSION="sd15_25_75_steps=100_dr_${CURRICULUM}${PSEUDO_SUFFIX}${POLICY_SUFFIX}"

echo "Running: train 25/75 split 5k datapoints with DR, $PSEUDO_TYPE, $CURRICULUM curriculum weight, using $POLICY_SOURCE policy for pseudo labels (DRST)"

# Build the command with optional flags
HARD_FLAG=""
if [ "$HARD_PSEUDO_LABEL" = "true" ]; then
    HARD_FLAG="--hard_pseudo_label"
fi

UPDATED_FLAG=""
if [ "$USE_UPDATED_POLICY" = "true" ]; then
    UPDATED_FLAG="--use_updated_policy_for_pseudo_labels"
fi

# accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr.py \
#     --mixed_precision "fp16" \
#     --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
#     --output_dir "$OUTPUT_DIR" \
#     --labeled_manifest "datasets/manifest/25_75/labeled.json" \
#     --unlabeled_manifest "datasets/manifest/25_75/unlabeled.json" \
#     --train_data_dir "datasets/pickapic_v2/" \
#     --train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --max_train_steps 100 \
#     --lr_warmup_steps 10 \
#     --learning_rate 1e-7 \
#     --scale_lr \
#     --checkpointing_steps 10000 \
#     --beta_dpo 5000 \
#     --report_to "wandb" \
#     --dataloader_num_workers 4 \
#     --mu 3 \
#     --curriculum $CURRICULUM \
#     $HARD_FLAG \
#     $UPDATED_FLAG

echo "Running: test 25/75 split 5k datapoints with DR, $PSEUDO_TYPE, $CURRICULUM curriculum weight, using $POLICY_SOURCE policy for pseudo labels (DRST)"
accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
    --prompts_path datasets/eval_prompts/pickapic_test.json \
    --model-path "$OUTPUT_DIR" \
    --version "$OUTPUT_VERSION" \
    --dataset pickscore \
    --reward_type pickscore \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --output-dir "test_outputs" \
    --num_imgs_per_prompt 4 \
    --batch_size 32 \
    --num_inference_steps 20 \
    --overwrite 0
