#!/usr/bin/env bash
set -u
# (Avoid set -e here because we manage background job failures manually.)

LIST_MODELS=(
  # training_runs/rebuttal/noise_flip/10_percent/checkpoint-100/unet
  # training_runs/rebuttal/noise_flip/30_percent/checkpoint-100/unet
  # training_runs/rebuttal/noise_flip/40_percent/checkpoint-100/unet
  # training_runs/rebuttal/noise_flip/50_percent/checkpoint-100/unet

  # training_runs/rebuttal/ablate_ratio/50_50/checkpoint-100/unet
  # training_runs/rebuttal/ablate_ratio/75_25/checkpoint-100/unet

  # training_runs/rebuttal/increase_labeled_size/100_4900/checkpoint-100/unet
  # training_runs/rebuttal/increase_labeled_size/200_4800/checkpoint-100/unet
  # training_runs/rebuttal/increase_labeled_size/500_4500/checkpoint-100/unet

  # training_runs/rebuttal/small_synthetic_portion/0.1_percent/checkpoint-100/unet
  # training_runs/rebuttal/small_synthetic_portion/0.05_percent/checkpoint-100/unet
  # "training_runs_common/ablate_ratio/dpo_50_50/checkpoint-100/unet"
  # "training_runs_common/ablate_ratio/dpo_75_25/checkpoint-100/unet"
  # "training_runs_common/increase_labeled_size/dpo_100/checkpoint-100/unet"
  # "training_runs_common/increase_labeled_size/dpo_200/checkpoint-100/unet"
  # "training_runs_common/increase_labeled_size/dpo_500/checkpoint-100/unet"
  # "training_runs_common/increase_labeled_size/dpo_1250/checkpoint-100/unet"
  # "training_runs_common/noise_flip/10_percent/dpo/checkpoint-100/unet"
  # "training_runs_common/noise_flip/20_percent/dpo/checkpoint-100/unet"
  # "training_runs_common/noise_flip/30_percent/dpo/checkpoint-100/unet"
  # "training_runs_common/noise_flip/40_percent/dpo/checkpoint-100/unet"
  # "training_runs_common/noise_flip/50_percent/dpo/checkpoint-100/unet"
  "training_runs_common/dpo_5k_25_75_animated/checkpoint-100/unet"
  "training_runs/dr_5k_25_75_animated/checkpoint-100/unet"
)

# ---- Config (copied from your L22-L48 script) ----
MODEL_PATH="checkpoints/sd15"
VERSION="sd15"
GEN_TYPE="pickapic_test"
NOISE_PATH="datasets/partiprompts_noise_sd15.pt"
JSON_PATH="datasets/eval_prompts/partiprompts.json"
BATCH_SIZE=8
GUIDANCE_SCALE=7.5
INFERENCE_STEPS=20
SPLIT_IDX=800

SAVE_ROOT="./main_results/rebuttal_dpo"
mkdir -p "$SAVE_ROOT"

# Use 8 GPUs -> 4 pairs. Edit if your machine uses different IDs.
GPUS=(0 1 2 3 4 5 6 7)
PAIRS=(
  "${GPUS[0]} ${GPUS[1]}"
  "${GPUS[2]} ${GPUS[3]}"
  "${GPUS[4]} ${GPUS[5]}"
  # "${GPUS[6]} ${GPUS[7]}"
)

# ---- Scheduler state ----
declare -A PID_TO_PAIR=()     # pid -> pair_index
declare -A PID_TO_DESC=()     # pid -> string (for logging)
declare -a PAIR_DONE_COUNT=(0 0 0 0)  # how many shards finished for each pair (0..2)
declare -a PAIR_BUSY=(0 0 0 0)        # 0 free, 1 busy

next_model_idx=0
total_models=${#LIST_MODELS[@]}

# Find a free pair index, echo it or echo -1 if none
get_free_pair() {
  for i in "${!PAIR_BUSY[@]}"; do
    if [[ "${PAIR_BUSY[$i]}" -eq 0 ]]; then
      echo "$i"
      return
    fi
  done
  echo -1
}

# Compute save_dir: ./main_results/rebuttal/<term3>/<term4>
# Example: training_runs/rebuttal/noise_flip/10_percent/checkpoint-100/unet
# -> ./main_results/rebuttal/noise_flip/10_percent
compute_save_dir() {
  local unet_path="$1"
  IFS='/' read -r t1 t2 t3 t4 rest <<< "$unet_path"
  echo "${SAVE_ROOT}/${t2}"
}

launch_checkpoint_on_pair() {
  local unet_path="$1"
  local pair_idx="$2"
  local gpuA gpuB
  read -r gpuA gpuB <<< "${PAIRS[$pair_idx]}"

  local save_dir
  save_dir="$(compute_save_dir "$unet_path")"
  mkdir -p "$save_dir"

  echo ">> Launching: unet_path=${unet_path}"
  echo "   save_dir=${save_dir}"
  echo "   GPUs: ${gpuA}, ${gpuB}"

  # Mark pair busy and reset done count
  PAIR_BUSY[$pair_idx]=1
  PAIR_DONE_COUNT[$pair_idx]=0

  # Shard 0: [0, SPLIT_IDX)
  CUDA_VISIBLE_DEVICES="$gpuA" python generate.py \
    gen_type="$GEN_TYPE" \
    model_path="$MODEL_PATH" \
    unet_path="$unet_path" \
    save_dir="$save_dir" \
    version="$VERSION" \
    noise_path="$NOISE_PATH" \
    json_path="$JSON_PATH" \
    batch_size="$BATCH_SIZE" \
    start_idx=0 \
    guidance_scale="$GUIDANCE_SCALE" \
    inference_steps="$INFERENCE_STEPS" \
    end_idx="$SPLIT_IDX" &

  local pid0=$!
  PID_TO_PAIR[$pid0]="$pair_idx"
  PID_TO_DESC[$pid0]="$(basename "$unet_path") shard0 gpu${gpuA} save_dir=${save_dir}"

  # Shard 1: [SPLIT_IDX, end)
  CUDA_VISIBLE_DEVICES="$gpuB" python generate.py \
    gen_type="$GEN_TYPE" \
    model_path="$MODEL_PATH" \
    unet_path="$unet_path" \
    save_dir="$save_dir" \
    version="$VERSION" \
    noise_path="$NOISE_PATH" \
    json_path="$JSON_PATH" \
    batch_size="$BATCH_SIZE" \
    start_idx="$SPLIT_IDX" \
    guidance_scale="$GUIDANCE_SCALE" \
    inference_steps="$INFERENCE_STEPS" \
    end_idx=-1 &

  local pid1=$!
  PID_TO_PAIR[$pid1]="$pair_idx"
  PID_TO_DESC[$pid1]="$(basename "$unet_path") shard1 gpu${gpuB} save_dir=${save_dir}"
}

# ---- Main loop ----
# Fill all pairs initially (up to 4 checkpoints)
while [[ $next_model_idx -lt $total_models ]]; do
  pair_idx="$(get_free_pair)"
  if [[ "$pair_idx" -eq -1 ]]; then
    break
  fi
  launch_checkpoint_on_pair "${LIST_MODELS[$next_model_idx]}" "$pair_idx"
  next_model_idx=$((next_model_idx + 1))
done

# Wait for jobs; when a pair completes both shards, launch the next checkpoint on it.
while :; do
  # If no running jobs and no remaining models, we are done
  if [[ ${#PID_TO_PAIR[@]} -eq 0 && $next_model_idx -ge $total_models ]]; then
    echo ">> All checkpoints finished."
    break
  fi

  # Wait for any one background job to finish
  wait -n
  status=$?

  # Identify which PID(s) ended: bash doesn't directly tell; we detect by checking 'jobs -pr'
  # We'll find finished PIDs by comparing our map against currently running pids.
  running_pids="$(jobs -pr | tr '\n' ' ')"

  for pid in "${!PID_TO_PAIR[@]}"; do
    if [[ " $running_pids " != *" $pid "* ]]; then
      pair_idx="${PID_TO_PAIR[$pid]}"
      desc="${PID_TO_DESC[$pid]}"

      if [[ $status -ne 0 ]]; then
        echo "!! Job failed (exit=$status): PID $pid :: $desc"
        exit $status
      else
        echo "<< Finished: PID $pid :: $desc"
      fi

      unset PID_TO_PAIR[$pid]
      unset PID_TO_DESC[$pid]

      PAIR_DONE_COUNT[$pair_idx]=$((PAIR_DONE_COUNT[$pair_idx] + 1))

      # If both shards finished, free the pair and launch the next checkpoint (if any)
      if [[ "${PAIR_DONE_COUNT[$pair_idx]}" -ge 2 ]]; then
        PAIR_BUSY[$pair_idx]=0
        PAIR_DONE_COUNT[$pair_idx]=0

        if [[ $next_model_idx -lt $total_models ]]; then
          launch_checkpoint_on_pair "${LIST_MODELS[$next_model_idx]}" "$pair_idx"
          next_model_idx=$((next_model_idx + 1))
        fi
      fi
    fi
  done
done
