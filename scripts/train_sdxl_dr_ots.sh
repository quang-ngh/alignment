SDXL_CKPT="/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
LR=2e-8
warmup_steps=5
rule="1:200,0.1"
lr_scheduler="piecewise_constant"
PROMPT_DIR="datasets/precomputed_prompt_embeds/5k_high_margin_sorted"
TRAIN_DATA_DIR="datasets/FiFA-100k-sorted/data/train"
LABEL_MANIFEST="datasets/manifest_high_margin/from_5k_high_margin_25_75/labeled.json"
UNLABEL_MANIFEST="datasets/manifest_high_margin/from_5k_high_margin_25_75/unlabeled.json"
LABEL_PSEUDO_MANIFEST="datasets/manifest_high_margin/from_5k_high_margin_25_75/pseudo_labeled_clip.json"
UNLABEL_PSEUDO_MANIFEST="datasets/manifest_high_margin/from_5k_high_margin_25_75/pseudo_unlabeled_clip.json"

accelerate launch --config-file "configs/train_sdxl_dr_ddp.yaml" train_sdxl_dpo_dr_ots.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path $SDXL_CKPT \
    --output_dir "training_runs/sdxl_dr_ots_high_margin_pseudo_clip" \
    --labeled_manifest $LABEL_MANIFEST \
    --unlabeled_manifest $UNLABEL_MANIFEST \
    --labeled_pseudo_manifest $LABEL_PSEUDO_MANIFEST \
    --unlabeled_pseudo_manifest $UNLABEL_PSEUDO_MANIFEST \
    --train_data_dir $TRAIN_DATA_DIR \
    --prompt_dir $PROMPT_DIR \
    --train_batch_size 1 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 64 \
    --max_train_steps 100 \
    --learning_rate $LR \
    --lr_warmup_steps $warmup_steps \
    --lr_scheduler $lr_scheduler \
    --lr_scheduler_rule $rule \
    --num_train_epochs 5 \
    --checkpointing_steps 50 \
    --gradient_checkpointing \
    --report_to "wandb" \
    --tracker_project_name "sdxl_dpo_dr" \
    --mu 1.0 \
    --scale_lr \
    --curriculum none \
