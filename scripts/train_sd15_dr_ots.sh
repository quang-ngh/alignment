LR=1e-7
warmup_steps=10
rule="1:200,0.25:400,0.1"
lr_scheduler="piecewise_constant"

LABEL_MANIFEST="datasets/manifest_high_margin/from_5k_high_margin_25_75/labeled.json"
LABEL_PSEUDO_MANIFEST="datasets/manifest_high_margin/from_5k_high_margin_25_75/pseudo_labeled_qwen.json"
UNLABEL_MANIFEST="datasets/manifest_high_margin/from_5k_high_margin_25_75/unlabeled.json"
UNLABEL_PSEUDO_MANIFEST="datasets/rebuttal/noise_flip/20_percent/pseudo_unlabeled.json"

SAVE_DIR="/common/users/hn315/training_runs/rebuttal_cvpr/noise_flip/20_percent_fix"
TRAIN_DATA_DIR="datasets/FiFA-100k-sorted/data/train"

accelerate launch --config-file "configs/sft.yaml" train_sd15_dpo_dr_ots.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "./checkpoints/sd15" \
    --output_dir $SAVE_DIR \
    --labeled_manifest $LABEL_MANIFEST \
    --unlabeled_manifest $UNLABEL_MANIFEST \
    --labeled_pseudo_manifest $LABEL_PSEUDO_MANIFEST \
    --unlabeled_pseudo_manifest $UNLABEL_PSEUDO_MANIFEST \
    --train_data_dir $TRAIN_DATA_DIR \
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
    --checkpointing_steps 100 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --tracker_project_name "rebuttal-cvpr" \
    --mu 1.0 \
    --scale_lr \
    --curriculum none \
