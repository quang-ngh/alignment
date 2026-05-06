#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/research/cbim/vast/hn315/alignment"
PYTHON_BIN="/research/cbim/vast/hn315/miniconda3/envs/adobe/bin/python"

MODEL_PATH="/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
UNET_PATH="/path/to/<unet_name>"

PROMPT_GLOB="datasets/eval_prompts/hpsv2_*.json"
OUTPUT_ROOT="qual"

BATCH_SIZE=32
GUIDANCE_SCALE=7.5
INFERENCE_STEPS=20
HEIGHT=1024
WIDTH=1024

NUM_SHARDS=4

SEEDS=(
    20770
    8296
    95385
    56302
    70281
)

cd "$REPO_ROOT"

for SEED in "${SEEDS[@]}"; do
    echo "Running seed ${SEED} with ${NUM_SHARDS} GPUs"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$REPO_ROOT/generate_sdxl_qual.py" \
        --model_path "$MODEL_PATH" \
        --unet_path "$UNET_PATH" \
        --seed "$SEED" \
        --prompt_glob "$PROMPT_GLOB" \
        --output_root "$OUTPUT_ROOT" \
        --batch_size "$BATCH_SIZE" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --inference_steps "$INFERENCE_STEPS" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --num_shards "$NUM_SHARDS" \
        --shard_index 0 &

    CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$REPO_ROOT/generate_sdxl_qual.py" \
        --model_path "$MODEL_PATH" \
        --unet_path "$UNET_PATH" \
        --seed "$SEED" \
        --prompt_glob "$PROMPT_GLOB" \
        --output_root "$OUTPUT_ROOT" \
        --batch_size "$BATCH_SIZE" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --inference_steps "$INFERENCE_STEPS" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --num_shards "$NUM_SHARDS" \
        --shard_index 1 &

    CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" "$REPO_ROOT/generate_sdxl_qual.py" \
        --model_path "$MODEL_PATH" \
        --unet_path "$UNET_PATH" \
        --seed "$SEED" \
        --prompt_glob "$PROMPT_GLOB" \
        --output_root "$OUTPUT_ROOT" \
        --batch_size "$BATCH_SIZE" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --inference_steps "$INFERENCE_STEPS" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --num_shards "$NUM_SHARDS" \
        --shard_index 2 &

    CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" "$REPO_ROOT/generate_sdxl_qual.py" \
        --model_path "$MODEL_PATH" \
        --unet_path "$UNET_PATH" \
        --seed "$SEED" \
        --prompt_glob "$PROMPT_GLOB" \
        --output_root "$OUTPUT_ROOT" \
        --batch_size "$BATCH_SIZE" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --inference_steps "$INFERENCE_STEPS" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --num_shards "$NUM_SHARDS" \
        --shard_index 3 &

    wait
done
