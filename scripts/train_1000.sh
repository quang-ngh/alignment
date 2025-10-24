#############################################################################################
# train 5k datapoints (100%)
# accelerate launch --gpu_ids 0,1 --num_processes=2 train_sd15_dpo.py \
#     --mixed_precision "bf16" \
#     --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
#     --output_dir "training_runs/sd15_5k_steps=1000" \
#     --labeled_manifest "datasets/manifest/from_5k/labeled.json" \
#     --unlabeled_manifest "datasets/manifest/from_5k/unlabeled.json" \
#     --train_data_dir "datasets/FiFA-100k/data/train" \
#     --train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_train_steps 1000 \
#     --lr_warmup_steps 10 \
#     --learning_rate 1e-7 \
#     --scale_lr \
#     --checkpointing_steps 100 \
#     --beta_dpo 5000 \
#     --report_to "wandb"

# for step in {100..1000..100}; do
#     echo "Testing at step ${step}"
#     accelerate launch --gpu_ids 0,1 --num_processes=2 --mixed_precision=bf16 fifa_test.py \
#         --prompts_path fifa_test_data/qas_test_filtered.json \
#         --model-path "training_runs/sd15_5k_steps=1000/checkpoint-${step}" \
#         --version "sd15_5k_steps=1000_checkpoint_${step}" \
#         --dataset pickscore \
#         --reward_type pickscore \
#         --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
#         --output-dir "test_outputs" \
#         --num_imgs_per_prompt 4 \
#         --overwrite 0
# done
#############################################################################################
# train 2.5k datapoints (50%)
accelerate launch --gpu_ids 2,3 --num_processes=2 train_sd15_dpo_simple.py \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --output_dir "training_runs/sd15_5k_steps=1000_50pc" \
    --manifest "datasets/manifest/from_5k/labeled.json" \
    --train_data_dir "datasets/FiFA-100k/data/train" \
    --train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 1000 \
    --lr_warmup_steps 10 \
    --learning_rate 1e-7 \
    --scale_lr \
    --checkpointing_steps 100 \
    --beta_dpo 5000 \
    --report_to "wandb"

for step in {100..1000..100}; do
    echo "Testing at step ${step}"
    accelerate launch --gpu_ids 2,3 --num_processes=2 --mixed_precision=bf16 fifa_test.py \
        --prompts_path fifa_test_data/qas_test_filtered.json \
        --model-path "training_runs/sd15_5k_steps=1000_50pc/checkpoint-${step}" \
        --version "sd15_5k_steps=1000_50pc_checkpoint_${step}" \
        --dataset pickscore \
        --reward_type pickscore \
        --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
        --output-dir "test_outputs" \
        --num_imgs_per_prompt 4 \
        --overwrite 0
done
#############################################################################################
# train 5k datapoints (100%) with pseudo labels for unlabeled data
# accelerate launch --gpu_ids 0,1 --num_processes=2 train_sd15_dpo.py \
#     --mixed_precision "bf16" \
#     --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
#     --output_dir "training_runs/sd15_5k_steps=1000_pseudo" \
#     --labeled_manifest "datasets/manifest/from_5k/labeled.json" \
#     --unlabeled_manifest "datasets/manifest/from_5k/unlabeled.json" \
#     --train_data_dir "datasets/FiFA-100k/data/train" \
#     --train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_train_steps 1000 \
#     --lr_warmup_steps 10 \
#     --learning_rate 1e-7 \
#     --scale_lr \
#     --checkpointing_steps 100 \
#     --beta_dpo 5000 \
#     --report_to "wandb" \
#     --use_pseudo_for_unlabeled \
#     --mu 1.0 \
#############################################################################################
# train with DR, soft pseudo labels and quadratic curriculum weight (DRST)
# accelerate launch --gpu_ids 2,3 --num_processes=2 train_sd15_dpo_dr.py \
#     --mixed_precision "bf16" \
#     --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
#     --output_dir "training_runs/sd15_5k_steps=1000_dr" \
#     --labeled_manifest "datasets/manifest/from_5k/labeled.json" \
#     --unlabeled_manifest "datasets/manifest/from_5k/unlabeled.json" \
#     --train_data_dir "datasets/FiFA-100k/data/train" \
#     --train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_train_steps 1000 \
#     --lr_warmup_steps 10 \
#     --learning_rate 1e-7 \
#     --scale_lr \
#     --checkpointing_steps 100 \
#     --beta_dpo 5000 \
#     --report_to "wandb" \
#     --mu 1.0 \
#     --curriculum quadratic \
#############################################################################################