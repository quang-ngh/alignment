MODEL_PATH="/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
LR=2.5e-8
warmup_steps=5
lr_scheduler="constant"

accelerate launch --config-file "configs/train_sdxl_dpo.yaml" train_sdxl_dpo_base.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path $MODEL_PATH \
    --output_dir "training_runs/sdxl_dpo_fifa5k_high_margin_100" \
    --good_manifest "datasets/manifest/5k_high_margin.json" \
    --train_data_dir "datasets/FiFA-100k/data/train" \
    --prompt_dir "datasets/precomputed_prompt_embeds/5k_high_margin" \
    --train_batch_size 1 \
    --dataloader_num_workers 32 \
    --gradient_accumulation_steps 32 \
    --max_train_steps 10000 \
    --learning_rate $LR \
    --lr_warmup_steps $warmup_steps \
    --lr_scheduler $lr_scheduler \
    --max_train_steps 1000 \
    --num_train_epochs 5 \
    --checkpointing_steps 100 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --tracker_project_name "dpo_fifa_5k" \
    --train_method "dpo" \
    # --lr_scheduler_rule $rule \
