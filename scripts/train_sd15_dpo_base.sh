LR=1e-7
warmup_steps=10
rule="1:200,0.25:400,0.1"
lr_scheduler="piecewise_constant"

accelerate launch --config-file "configs/train_multigpu.yaml" train_sd15_dpo_base.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "./checkpoints/sd15" \
    --output_dir "training_runs/sd15_dpo_base_new_data" \
    --good_manifest "datasets/manifest/fifa_new.json" \
    --train_data_dir "datasets/FiFA-new/data/train" \
    --train_batch_size 8 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 1000 \
    --learning_rate $LR \
    --lr_warmup_steps $warmup_steps \
    --lr_scheduler $lr_scheduler \
    --lr_scheduler_rule $rule \
    --max_train_steps 1000 \
    --num_train_epochs 5 \
    --checkpointing_steps 200 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --tracker_project_name "sd15_dpo_base_new_data" \
    --train_method "dpo" \
