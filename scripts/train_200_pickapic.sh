#!/bin/bash

# Usage: ./train_500_pickapic.sh [command_name] [gpu_ids]
# Available commands:
#   5k - Train 5k datapoints
#   1k - Train 1k datapoints
#   pseudo_5k - Train 5k datapoints with pseudo labels for unlabeled data
#   dr_5k - Train 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight

# Configuration: Modify these variables to set output directories
TRAIN_OUTPUT_BASE_DIR="train_outputs_dr_v1"  # Base directory for training outputs
TEST_OUTPUT_DIR="test_outputs_dr_v1"          # Directory for test outputs

COMMAND=${1:-"5k"}
GPU_IDS=${2:-"0,1"}

# Calculate num_processes from GPU IDs (count commas + 1)
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "Using GPU IDs: $GPU_IDS (num_processes=$NUM_PROCESSES)"
echo "Using training base dir: $TRAIN_OUTPUT_BASE_DIR"
echo "Using test output dir: $TEST_OUTPUT_DIR"

# Generate random port for distributed training (avoid port conflicts)
MASTER_PORT=$((29500 + RANDOM % 100))
export MASTER_PORT
echo "Using MASTER_PORT: $MASTER_PORT"

# Shared arguments - common to all training commands
SHARED_ARGS="--mixed_precision fp16 --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 --train_data_dir datasets/pickapicv2_fifa/train --max_train_steps 200 --lr_warmup_steps 10 --learning_rate 1e-7 --scale_lr --checkpointing_steps 100 --beta_dpo 5000 --report_to wandb --dataloader_num_workers 4 --lr_scheduler piecewise_constant --lr_scheduler_rule 1:200,0.25:400,0.1 --resume_from_checkpoint latest"

# Shared arguments for DR-based commands
DR_ARGS="--mu 1.0 --threshold 0.0"

# Shared arguments for external model commands (qwen/clip)
LABELED_MANIFEST="datasets/pickapicv2_fifa/manifest/5k/labeled.json"
UNLABELED_MANIFEST="datasets/pickapicv2_fifa/manifest/5k/labeled.json"

case "$COMMAND" in
    "5k")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 5k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_simple.py \
            $SHARED_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --manifest "datasets/hpdv2_fifa/manifest/last_5k/hpdv2_fifa_5k.json" \
            --train_batch_size 8 \
            --gradient_accumulation_steps 8
        ;;
    
    "1.25k")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_1.25k_steps=200"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 1.25k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_simple.py \
            $SHARED_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --manifest "$LABELED_MANIFEST" \
            --train_batch_size 8 \
            --gradient_accumulation_steps 8
        ;;
    
    "pseudo_25_75")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels for unlabeled data"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --no-dr
        ;;
    
    "dr_25_75")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_dr"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with DR self-training, hard pseudo labels and no curriculum"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none
        ;;
    
    "dr_25_75_qwen")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_dr_qwen"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from Qwen, DR"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "qwen" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none
        ;;

    "pseudo_25_75_qwen")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_qwen"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from Qwen, no DR"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "qwen" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none \
            --no-dr
        ;;

    
    "dr_25_75_clip")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_dr_clip"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from CLIP, DR"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "clip" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none
        ;;

    "pseudo_25_75_clip")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_clip"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from CLIP, no DR"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "clip" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none \
            --no-dr
        ;;
    "dr_25_75_smolvlm")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_dr_smolvlm"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from SMOLVLM, DR"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "smolvlm" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none        
        ;;
    "pseudo_25_75_smolvlm")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_smolvlm"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from SMOLVLM, no DR"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "smolvlm" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none \
            --no-dr
        ;;

    "pseudo_25_75_qwen_dr_iclr25")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_qwen_dr_iclr25"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from qwen, DR-ICLR25"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_robust_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "qwen" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --robustness_method "dr_iclr25"
        ;;
    "pseudo_25_75_qwen_label_smoothing")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_qwen_label_smoothing"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from qwen, label smoothing"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_robust_external.py \
            $SHARED_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "qwen" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --robustness_method "label_smoothing" \
            --noise_ratio 0.2
        ;;
    "pseudo_25_75_qwen_ipo")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_qwen_ipo"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from qwen, IPO"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_robust_external.py \
            $SHARED_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "qwen" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --robustness_method "ipo" \
            --beta_dpo 100
        ;;
    "pseudo_25_75_noise_dr_iclr25")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_noise_dr_iclr25"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from noise, DR-ICLR25"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_robust_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "noise" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --robustness_method "dr_iclr25"
        ;;
    "pseudo_25_75_noise_label_smoothing")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_noise_label_smoothing"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from noise, label smoothing"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_robust_external.py \
            $SHARED_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "noise" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --robustness_method "label_smoothing" \
            --noise_ratio 0.2
        ;;
    "pseudo_25_75_noise_ipo")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_last_5k_steps=200_pseudo_noise_ipo"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels from noise, IPO"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_robust_external.py \
            $SHARED_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "$LABELED_MANIFEST" \
            --unlabeled_manifest "$UNLABELED_MANIFEST" \
            --external_model "noise" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --robustness_method "ipo"
        ;;
    "dr_50_50_qwen")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_5k_50_50_steps=200_dr_qwen"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 50/50 split 5k datapoints with pseudo labels from Qwen, DR"
        accelerate launch --config_file drdpo_config.yaml --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr_external.py \
            $SHARED_ARGS \
            $DR_ARGS \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest datasets/pickapicv2_fifa/manifest/5k-50-50/labeled.json \
            --unlabeled_manifest datasets/pickapicv2_fifa/manifest/5k-50-50/unlabeled.json  \
            --external_model "qwen" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --curriculum none
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        echo "Usage: ./train_200_pickapic.sh [command_name] [gpu_ids]"
        echo ""
        echo "Examples:"
        echo "  ./train_200_pickapic.sh pseudo_5k"
        echo "  ./train_200_pickapic.sh pseudo_5k 0,1"
        echo "  ./train_200_pickapic.sh dr_5k 2,3"
        echo ""
        echo "Available commands:"
        echo "  5k        - Train 5k datapoints using simple script"
        echo "  1.25k        - Train 1.25k datapoints using simple script"
        echo "  pseudo_5k - Train 5k datapoints with pseudo labels for unlabeled data"
        echo "  dr_5k     - Train 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight"
        exit 1
        ;;
esac
