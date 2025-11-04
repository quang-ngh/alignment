#!/bin/bash

# Usage: ./test.sh [gpu_ids] <model-path> [output-dir] [dataset]
#   gpu_ids: Comma-separated GPU IDs (default: "2,3")
#   model-path: Path to the model directory (e.g., "train_outputs_dr_v1/sd15_25_75_steps=100_dr_curriculum=none_threshold=0.")
#   output-dir: Output directory for test results (default: "test_outputs_dr_v1")
#   dataset: Dataset name - pickscore or partiprompts (default: "partiprompts")

GPU_IDS=${1:-"2,3"}
MODEL_PATH=$2
OUTPUT_DIR=${3:-"test_outputs_dr_v1"}
DATASET=${4:-"partiprompts"}
SDXL=${5:-"0"}

if [ -z "$MODEL_PATH" ]; then
    echo "Error: model-path is required"
    echo "Usage: ./test.sh [gpu_ids] <model-path> [output-dir] [dataset]"
    echo "Example: ./test.sh 2,3 train_outputs_dr_v1/sd15_25_75_steps=100_dr_curriculum=none_threshold=0. test_outputs_dr_v1 partiprompts"
    echo "Supported datasets: pickapic, partiprompts"
    exit 1
fi

# Map dataset names to prompt file paths
case "$DATASET" in
    pickapic)
        PROMPTS_PATH="datasets/eval_prompts/pickapic_test.json"
        ;;
    partiprompts)
        PROMPTS_PATH="datasets/eval_prompts/qas_parti_test.json"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Supported datasets: pickapic, partiprompts"
        exit 1
        ;;
esac

# Set pretrained model based on SDXL flag
if [ "$SDXL" = "1" ]; then
    PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
else
    PRETRAINED_MODEL="runwayml/stable-diffusion-v1-5"
fi

# Derive version from model-path
# If path ends with checkpoint-X, use model_name_checkpoint-X
# Otherwise, use just model_name
BASENAME=$(basename "$MODEL_PATH")
if [[ "$BASENAME" =~ ^checkpoint- ]]; then
    # Path ends with checkpoint-X, get parent directory name
    MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    CHECKPOINT_NAME="$BASENAME"
    VERSION="${MODEL_NAME}_${CHECKPOINT_NAME}"
else
    # Path ends with model directory
    VERSION="$BASENAME"
fi

# Calculate num_processes from GPU IDs (count commas + 1)
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Generate a random port for distributed training (29500-30000 range)
RANDOM_PORT=$((29500 + RANDOM % 500))

echo "Model path: $MODEL_PATH"
echo "Version (derived): $VERSION"
echo "GPU IDs: $GPU_IDS (num_processes=$NUM_PROCESSES)"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Prompts path: $PROMPTS_PATH"
echo "Pre-trained model: $PRETRAINED_MODEL"
echo "Using random port: $RANDOM_PORT"

accelerate launch --gpu_ids "$GPU_IDS" --num_processes=$NUM_PROCESSES --main_process_port=$RANDOM_PORT \
    --mixed_precision=fp16 fifa_test.py \
    --prompts_path "$PROMPTS_PATH" \
    --model-path "$MODEL_PATH" \
    --version "$VERSION" \
    --dataset "$DATASET" \
    --reward_type pickscore \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --num_imgs_per_prompt 1 \
    --batch_size 16 \
    --num_inference_steps 20 \
    --overwrite 0