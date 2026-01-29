LR=1e-7
warmup_steps=10
rule="1:200,0.25:400,0.1"
lr_scheduler="piecewise_constant"

MANIFEST="datasets/manifest_5k_animated/5k_25_75/dpo_concat.json"
CONFIG="configs/sft.yaml"
OUTPUT_DIR="/common/users/hn315/training_runs/cvpr_rebuttal/dpo_5k_25_75_animated"
TRAIN_DATA_DIR="datasets/pickapic-5k-animated-flipped-30pct/data/train"

accelerate launch --config-file $CONFIG train_sd15_dpo_base.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "./checkpoints/sd15" \
    --output_dir $OUTPUT_DIR \
    --good_manifest $MANIFEST \
    --train_data_dir $TRAIN_DATA_DIR \
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
    --tracker_project_name "cvpr_rebuttal" \
    --train_method "dpo" \
    --scale_lr \
