LR=1e-7
warmup_steps=10
rule="1:200,0.25:400,0.1"
lr_scheduler="piecewise_constant"

LABEL_MANIFEST="datasets/manifest_high_margin/from_50k_high_margin_25_75/labeled.json"
UNLABEL_MANIFEST="datasets/manifest_high_margin/from_50k_high_margin_25_75/unlabeled.json"
LABEL_PSEUDO_MANIFEST="datasets/manifest_high_margin/from_50k_high_margin_25_75/pseudo_labeled_qwen.json"
UNLABEL_PSEUDO_MANIFEST="datasets/manifest_high_margin/from_50k_high_margin_25_75/pseudo_unlabeled_qwen.json"
ACC_CONFIG="configs/ablation/50k.yaml"

accelerate launch --config-file $ACC_CONFIG train_sd15_dpo_dr_ots.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "./checkpoints/sd15" \
    --output_dir "training_runs/sd15_dr_ots_high_margin_pseudo_qwen_50k" \
    --labeled_manifest $LABEL_MANIFEST \
    --unlabeled_manifest $UNLABEL_MANIFEST \
    --labeled_pseudo_manifest $LABEL_PSEUDO_MANIFEST \
    --unlabeled_pseudo_manifest $UNLABEL_PSEUDO_MANIFEST \
    --train_data_dir "datasets/FiFA-100k-sorted/data/train" \
    --train_batch_size 8 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 3000 \
    --learning_rate $LR \
    --lr_warmup_steps $warmup_steps \
    --lr_scheduler $lr_scheduler \
    --lr_scheduler_rule $rule \
    --num_train_epochs 5 \
    --checkpointing_steps 100 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --tracker_project_name "sd15_dpo_dr" \
    --mu 1.0 \
    --scale_lr \
    --curriculum none \
