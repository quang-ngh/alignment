# CUDA_VISIBLE_DEVICES=1 python evaluator.py \
#     benchmark_type="hpsv2" \
#     image_dir="./main_results/hpsv2/sdxl_sft_5k_high_margin_checkpoint-10./main_results/hpsv2/sdxl_sft_5k_high_margin_checkpoint-100
#     prompt_dir="datasets/eval_prompts" \
#     name="eval_results/main/hpsv2/sdxl_sft_5k_high_margin_checkpoint-100"  &

# CUDA_VISIBLE_DEVICES=1 python evaluator.py \
#     benchmark_type="hpsv2" \
#     image_dir="main_results/hpsv2/sd15_dr_ots_high_margin_pseudo_clip" \
#     prompt_dir="datasets/eval_prompts" \
#     name="eval_results/main/hpsv2/sd15_dr_ots_high_margin_pseudo_clip_v2"  &

# CUDA_VISIBLE_DEVICES=6 python evaluator.py \
#     benchmark_type="hpsv2" \
#     image_dir="main_results/hpsv2/sd15_dr_dpo_38k_unl" \
#     prompt_dir="datasets/eval_prompts" \
#     name="eval_results/main/hpsv2/sd15_dr_dpo_38k_unl"

MODELS=(
    # "main_results/hpsv2/sd15_dpo_8k_unl"
    # "main_results/hpsv2/sd15_dpo_38k_unl"
    # "main_results/hpsv2/sd15_dpo_98k_unl"
    # "main_results/hpsv2/sd15_dr_dpo_8k_unl"
    # "main_results/hpsv2/sd15_dr_dpo_38k_unl"
    # "main_results/hpsv2/sd15_dr_dpo_98k_unl"
    # "main_results/hpsv2/sd15_dpo_smolvlm_pseudo_ckpt100"
    # "main_results/hpsv2/sd15_dr_dpo_pseudo_smolvlm_ckpt100"
    "main_results/hpsv2/sd15_dpo_flip80_pseudo_ckpt100"
    "main_results/hpsv2/sd15_dr_dpo_pseudo_flip80_ckpt100"
)

GPUS=(2 3)

for ((i=0; i<${#MODELS[@]}; i++)); do
    eval_model="${MODELS[$i]}"
    gpu_idx=$((i % 2))
    save_name="eval_results/main/hpsv2/$(basename ${MODELS[$i]})"
    echo "Evaluating $eval_model"
    echo "Saving to $save_name"
    CUDA_VISIBLE_DEVICES=${GPUS[$gpu_idx]} python evaluator.py \
        benchmark_type="hpsv2" \
        image_dir="$eval_model" \
        prompt_dir="datasets/eval_prompts" \
        name="$save_name" &
    # Wait every 2 jobs
    if (( (i+1) % 2 == 0 )); then
        wait
    fi
done
wait