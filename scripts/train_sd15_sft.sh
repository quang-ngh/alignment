LR=1e-7
warmup_steps=10
rule="1:200,0.25:400,0.1"
lr_scheduler="piecewise_constant"

# MANIFEST="datasets/manifest_high_margin/100k_labaled_pseudo_unlabled.json"
# CONFIG="configs/ablation/100k.yaml"
# OUTPUT_DIR="/common/users/hn315/alignment/ablate_dpo_100k_pseudo"
MANIFEST="datasets/manifest_high_margin/5k_high_margin.json"
CONFIG="configs/sft.yaml"
OUTPUT_DIR="/common/users/hn315/alignment/sft_5k_high_margin"


accelerate launch --config-file $CONFIG train_sd15_sft.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "./checkpoints/sd15" \
    --output_dir $OUTPUT_DIR \
    --manifest $MANIFEST \
    --train_data_dir "datasets/FiFA-100k-sorted/data/train" \
    --train_batch_size 8 \
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
    --tracker_project_name "sd15_sft" \
    --scale_lr \
