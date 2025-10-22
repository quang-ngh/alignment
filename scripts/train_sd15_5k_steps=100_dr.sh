accelerate launch --gpu_ids 2,3 --num_processes=2 train_sd15_dpo_dr.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --output_dir "training_runs/sd15_5k_steps=100_dr" \
    --labeled_manifest "datasets/manifest/from_5k/labeled.json" \
    --unlabeled_manifest "datasets/manifest/from_5k/unlabeled.json" \
    --train_data_dir "datasets/FiFA-100k/data/train" \
    --train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 100 \
    --lr_warmup_steps 10 \
    --learning_rate 1e-7 \
    --scale_lr \
    --checkpointing_steps 10000 \
    --beta_dpo 5000 \
    --report_to "wandb" \
    --mu 1.0 \
    # --gradient_checkpointing \