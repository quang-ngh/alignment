#!/bin/bash

# Usage: ./train_100.sh [command_name] [gpu_ids]
# Available commands:
#   5k - Train 5k datapoints
#   1.25k - Train 1.25k datapoints
#   pseudo_25_75 - Train 25/75 split 5k datapoints with pseudo labels for unlabeled data
#   drst_25_75 - Train 25/75 split 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight (DRST)

COMMAND=${1:-"pseudo_25_75"}
GPU_IDS=${2:-"0,1"}

# Calculate num_processes from GPU IDs (count commas + 1)
NUM_PROCESSES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "Using GPU IDs: $GPU_IDS (num_processes=$NUM_PROCESSES)"

case "$COMMAND" in
    "5k")
        echo "Running: train 5k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_simple.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "training_runs/sd15_5k_new_steps=100" \
            --manifest "datasets/manifest/fifa_5k_new.json" \
            --train_data_dir "datasets/pickapic_v2/" \
            --train_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --max_train_steps 100 \
            --lr_warmup_steps 10 \
            --learning_rate 1e-7 \
            --scale_lr \
            --checkpointing_steps 10000 \
            --beta_dpo 5000 \
            --report_to "wandb"
    
        echo "Running: test 5k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "training_runs/sd15_5k_new_steps=100" \
            --version "sd15_5k_new_steps=100" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "test_outputs" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    "1.25k")
        echo "Running: train 1.25k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_simple.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "training_runs/sd15_1.25k_steps=100" \
            --manifest "datasets/manifest/25_75/labeled.json" \
            --train_data_dir "datasets/pickapic_v2/" \
            --train_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --max_train_steps 100 \
            --lr_warmup_steps 10 \
            --learning_rate 1e-7 \
            --scale_lr \
            --checkpointing_steps 10000 \
            --beta_dpo 5000 \
            --report_to "wandb"
        echo "Running: test 1.25k datapoints"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "training_runs/sd15_1.25k_steps=100" \
            --version "sd15_1.25k_steps=100" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "test_outputs" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    
    "pseudo_25_75")
        echo "Running: train 25/75 split 5k datapoints with pseudo labels for unlabeled data"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "training_runs/sd15_25_75_steps=100_pseudo" \
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
            --use_pseudo_for_unlabeled \
            --mu 3
        echo "Running: test 25/75 split 5k datapoints with pseudo labels for unlabeled data"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "training_runs/sd15_25_75_steps=100_pseudo" \
            --version "sd15_25_75_steps=100_pseudo" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "test_outputs" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    
    "drst_25_75")
        echo "Running: train 25/75 split 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight (DRST)"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES train_sd15_dpo_dr.py \
            --mixed_precision "fp16" \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir "training_runs/sd15_25_75_steps=100_dr" \
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
            --curriculum quadratic
        echo "Running: test 25/75 split 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight (DRST)"
        accelerate launch --gpu_ids $GPU_IDS --num_processes=$NUM_PROCESSES fifa_test.py \
            --prompts_path datasets/eval_prompts/pickapic_test.json \
            --model-path "training_runs/sd15_25_75_steps=100_dr" \
            --version "sd15_25_75_steps=100_dr" \
            --dataset pickscore \
            --reward_type pickscore \
            --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output-dir "test_outputs" \
            --num_imgs_per_prompt 4 \
            --batch_size 32 \
            --num_inference_steps 20 \
            --overwrite 0
        ;;
    
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        echo "Usage: ./train_100.sh [command_name] [gpu_ids]"
        echo ""
        echo "Examples:"
        echo "  ./train_100.sh pseudo_25_75"
        echo "  ./train_100.sh pseudo_25_75 0,1"
        echo "  ./train_100.sh drst_25_75 2,3"
        echo ""
        echo "Available commands:"
        echo "  5k              - Train 5k datapoints using simple script"
        echo "  1.25k           - Train 1.25k datapoints using simple script"
        echo "  pseudo_25_75    - Train 25/75 split 5k datapoints with pseudo labels for unlabeled data"
        echo "  drst_25_75      - Train 25/75 split 5k datapoints with DR, soft pseudo labels and quadratic curriculum weight (DRST)"
        exit 1
        ;;
esac
