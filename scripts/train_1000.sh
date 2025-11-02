#!/bin/bash

# Usage: ./train_1000.sh [command_name] [gpu_ids]
# Available commands:
#   5k - Train 5k datapoints
#   1k - Train 1k datapoints
#   pseudo_5k - Train 5k datapoints with pseudo labels for unlabeled data
#   dr_5k - Train 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight

# Configuration: Modify these variables to set output directories
TRAIN_OUTPUT_BASE_DIR="train_outputs_new_env"  # Base directory for training outputs
TEST_OUTPUT_DIR="test_outputs_new_env"          # Directory for test outputs

COMMAND=${1:-"5k"}
GPU_IDS=${2:-"0,1"}

# Calculate num_processes from GPU IDs (count commas + 1)
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "Using GPU IDs: $GPU_IDS (num_processes=$NUM_PROCESSES)"
echo "Using training base dir: $TRAIN_OUTPUT_BASE_DIR"
echo "Using test output dir: $TEST_OUTPUT_DIR"

case "$COMMAND" in
    "5k")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_5k_steps=1000"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 5k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_simple.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "$TRAIN_OUTPUT" \
            --manifest "datasets/manifest/fifa_5k_new.json" \
            --train_data_dir "datasets/pickapic_v2/" \
            --train_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --max_train_steps 1000 \
            --lr_warmup_steps 10 \
            --learning_rate 1e-7 \
            --scale_lr \
            --checkpointing_steps 100 \
            --beta_dpo 5000 \
            --dataloader_num_workers 4 \
            --report_to "wandb" \
            --lr_scheduler "piecewise_constant" \
            --lr_scheduler_rule "1:200,0.25:400,0.1"
    
        echo "Running: test 5k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "$TRAIN_OUTPUT" \
            --version "$VERSION" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "$TEST_OUTPUT_DIR" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    
    "1.25k")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_1.25k_steps=1000"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 1.25k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_simple.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "$TRAIN_OUTPUT" \
            --manifest "datasets/manifest/25_75/labeled.json" \
            --train_data_dir "datasets/pickapic_v2/" \
            --train_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --max_train_steps 1000 \
            --lr_warmup_steps 10 \
            --learning_rate 1e-7 \
            --scale_lr \
            --checkpointing_steps 100 \
            --beta_dpo 5000 \
            --dataloader_num_workers 4 \
            --report_to "wandb" \
            --lr_scheduler "piecewise_constant" \
            --lr_scheduler_rule "1:200,0.25:400,0.1"
    
        echo "Running: test 1.25k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "$TRAIN_OUTPUT" \
            --version "$VERSION" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "$TEST_OUTPUT_DIR" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    
    "pseudo_25_75")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_25_75_steps=1000_pseudo"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with pseudo labels for unlabeled data"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "datasets/manifest/25_75/labeled.json" \
            --unlabeled_manifest "datasets/manifest/25_75/unlabeled.json" \
            --train_data_dir "datasets/pickapic_v2/" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --max_train_steps 1000 \
            --lr_warmup_steps 10 \
            --learning_rate 1e-7 \
            --scale_lr \
            --checkpointing_steps 100 \
            --beta_dpo 5000 \
            --report_to "wandb" \
            --use_pseudo_for_unlabeled \
            --mu 1.0
        echo "Running: test 25/75 split 5k datapoints with pseudo labels for unlabeled data"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "$TRAIN_OUTPUT" \
            --version "$VERSION" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "$TEST_OUTPUT_DIR" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    
    "drst_25_75")
        TRAIN_OUTPUT="$TRAIN_OUTPUT_BASE_DIR/sd15_25_75_steps=1000_drst"
        VERSION=$(basename "$TRAIN_OUTPUT")
        
        echo "Running: train 25/75 split 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight (DRST)"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "$TRAIN_OUTPUT" \
            --labeled_manifest "datasets/manifest/25_75/labeled.json" \
            --unlabeled_manifest "datasets/manifest/25_75/unlabeled.json" \
            --train_data_dir "datasets/pickapic_v2/" \
            --train_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --max_train_steps 1000 \
            --lr_warmup_steps 10 \
            --learning_rate 1e-7 \
            --scale_lr \
            --checkpointing_steps 100 \
            --beta_dpo 5000 \
            --report_to "wandb" \
            --mu 1.0 \
            --curriculum quadratic
        echo "Running: test 25/75 split 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight (DRST)"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "$TRAIN_OUTPUT" \
            --version "$VERSION" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "$TEST_OUTPUT_DIR" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        echo "Usage: ./train_1000.sh [command_name] [gpu_ids]"
        echo ""
        echo "Examples:"
        echo "  ./train_1000.sh pseudo_5k"
        echo "  ./train_1000.sh pseudo_5k 0,1"
        echo "  ./train_1000.sh dr_5k 2,3"
        echo ""
        echo "Available commands:"
        echo "  5k        - Train 5k datapoints using simple script"
        echo "  1k        - Train 1k datapoints using simple script"
        echo "  pseudo_5k - Train 5k datapoints with pseudo labels for unlabeled data"
        echo "  dr_5k     - Train 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight"
        exit 1
        ;;
esac
