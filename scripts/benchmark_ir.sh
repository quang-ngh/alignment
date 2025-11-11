results=(
    # "main_results/partiprompts/sd15_ablate_dpo_10k_pseudo"
    # "main_results/partiprompts/sd15_ablate_dpo_20k_pseudo"
    # "main_results/partiprompts/sd15_ablate_dpo_50k_pseudo"
    # "main_results/partiprompts/sd15_ablate_dpo_100k_pseudo"
    # "main_results/partiprompts/sdxl_dr_updated_policy"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-200"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-300"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-400"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-500"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-600"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-700"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-800"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-900"
    # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-1000"
    # "main_results/partiprompts/sd15_base"
    # "main_results/partiprompts/sd15_dpo_25label"
    # "main_results/partiprompts/sd15_dpo_100label"
    # "main_results/partiprompts/sd15_dpo_base_labeled_and_pseudo_unlabeled_clip"
    # "main_results/partiprompts/sd15_dpo_base_labeled_and_pseudo_unlabeled_qwen"
    # "main_results/partiprompts/sd15_dr_ots_clip"   
    "main_results/partiprompts/sd15_dr_ots_high_margin_pseudo_qwen"
    # "main_results/partiprompts/sd15_sft_5k_checkpoint-100"
)

for result in "${results[@]}"; do
    subfolder=$(echo "$result" | awk -F'/' '{print $3}')
    name="eval_results/main/pickscore/${subfolder}"
    echo "Evaluating $name"
    CUDA_VISIBLE_DEVICES=2 python evaluator.py \
        benchmark_type="ir" \
        image_dir=$result \
        prompt_dir="datasets/eval_prompts/partiprompts.json" \
        name="eval_results/main/ir/sd15_ablate_dpo_${subfolder}" &
done

# CUDA_VISIBLE_DEVICES=5 python evaluator.py \
#     benchmark_type="pickscore" \
#     image_dir="main_results/partiprompts/sdxl_dr_ots_high_margin_pseudo_clip" \
#     prompt_dir="datasets/eval_prompts/partiprompts.json" \
#     name="eval_results/main/pickscore/sdxl_dr_ots_high_margin_pseudo_clip" &

# CUDA_VISIBLE_DEVICES=2 python evaluator.py \
#     benchmark_type="pickscore" \
#     image_dir="main_results/partiprompts/sd15_dr_ots_high_margin_pseudo_qwen_20k" \
#     prompt_dir="datasets/eval_prompts/partiprompts.json" \
#     name="eval_results/main/pickscore/sd15_dr_ots_high_margin_pseudo_qwen_20k" &

# CUDA_VISIBLE_DEVICES=4 python evaluator.py \
#     benchmark_type="pickscore" \
#     image_dir="main_results/partiprompts/sd15_dpo_25label" \
#     prompt_dir="datasets/eval_prompts/partiprompts.json" \
#     name="eval_results/main/pickscore/sd15_dpo_25label" &

# CUDA_VISIBLE_DEVICES=5 python evaluator.py \
#     benchmark_type="pickscore" \
#     image_dir="main_results/partiprompts/sd15_dpo_100label" \
#     prompt_dir="datasets/eval_prompts/partiprompts.json" \
#     name="eval_results/main/pickscore/sd15_dpo_100label" &

