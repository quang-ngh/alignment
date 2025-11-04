LR=1e-7
warmup_steps=10
rule="1:200,0.25:400,0.1"
lr_scheduler="piecewise_constant"

accelerate launch --config-file "configs/train_multigpu.yaml" train_sd15_dpo_base.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "./checkpoints/sd15" \
    --output_dir "training_runs/sd15_dpo_base_labeled_and_pseudo_unlabeled_qwen" \
    --good_manifest "datasets/manifest_high_margin/labeled_and_pseudo_unlabeled_qwen.json" \
    --train_data_dir "datasets/FiFA-100k-sorted/data/train" \
    --train_batch_size 4 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 1000 \
    --learning_rate $LR \
    --lr_warmup_steps $warmup_steps \
    --lr_scheduler $lr_scheduler \
    --lr_scheduler_rule $rule \
    --num_train_epochs 5 \
    --checkpointing_steps 100 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --tracker_project_name "sd15_dpo_base_new_data" \
    --train_method "dpo" \
    --scale_lr \
