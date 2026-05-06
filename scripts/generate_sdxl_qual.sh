#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_MODEL_PATH="/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <unet_path> <seed> [model_path] [gpu_spec] [extra generate_sdxl_qual.py args...]"
    exit 1
fi

UNET_PATH="$1"
SEED="$2"
MODEL_PATH="${3:-$DEFAULT_MODEL_PATH}"
GPU_SPEC="${4:-}"

EXTRA_ARGS=()
if [[ $# -gt 4 ]]; then
    EXTRA_ARGS=("${@:5}")
fi

BATCH_SIZE="${BATCH_SIZE:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
PROMPT_GLOB="${PROMPT_GLOB:-datasets/eval_prompts/hpsv2_*.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-qual}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_ROOT"

if [[ -z "$GPU_SPEC" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPU_SPEC="$CUDA_VISIBLE_DEVICES"
    else
        GPU_SPEC="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    fi
fi

GPU_SPEC="${GPU_SPEC// /}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_SPEC"
NUM_SHARDS="${#GPU_ARRAY[@]}"

if [[ "$NUM_SHARDS" -lt 1 ]]; then
    echo "No GPUs found."
    exit 1
fi

echo "Using GPUs: ${GPU_ARRAY[*]}"
echo "Launching $NUM_SHARDS shard(s)"

for shard_index in "${!GPU_ARRAY[@]}"; do
    gpu_id="${GPU_ARRAY[$shard_index]}"
    echo "Launching shard $((shard_index + 1))/$NUM_SHARDS on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$REPO_ROOT/generate_sdxl_qual.py" \
        --model_path "$MODEL_PATH" \
        --unet_path "$UNET_PATH" \
        --seed "$SEED" \
        --prompt_glob "$PROMPT_GLOB" \
        --output_root "$OUTPUT_ROOT" \
        --batch_size "$BATCH_SIZE" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --inference_steps "$INFERENCE_STEPS" \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$shard_index" \
        "${EXTRA_ARGS[@]}" &
done

wait
