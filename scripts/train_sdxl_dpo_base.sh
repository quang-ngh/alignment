MODEL_PATH="/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
LR=2e-8
warmup_steps=5
rule="1:200,0.1"
lr_scheduler="piecewise_constant"
MANIFEST="datasets/manifest_high_margin/5k_labeled_and_pseudo_unlabeled_qwen.json"
OUTPUT_DIR="training_runs/sdxl_dpo_pseudo"

accelerate launch --config-file "configs/train_sdxl_dpo_ddp.yaml" train_sdxl_dpo_base.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --good_manifest $MANIFEST \
    --train_data_dir "datasets/FiFA-100k-sorted/data/train" \
    --prompt_dir "datasets/precomputed_prompt_embeds/5k_high_margin_sorted" \
    --train_batch_size 1 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 64 \
    --max_train_steps 100 \
    --learning_rate $LR \
    --lr_warmup_steps $warmup_steps \
    --lr_scheduler $lr_scheduler \
    --num_train_epochs 5 \
    --checkpointing_steps 50 \
    --report_to "wandb" \
    --tracker_project_name "dpo_fifa_5k" \
    --train_method "dpo" \
    --lr_scheduler_rule $rule \
    --scale_lr \
    --gradient_checkpointing \
