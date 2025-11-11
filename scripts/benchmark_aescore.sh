results=(
    # "main_results/partiprompts/sdxl_dpo_fifa5k_high_margin_100_v2"
    "main_results/partiprompts/sdxl_dpo_fifa5k_high_margin_25"
    # "main_results/partiprompts/sdxl_dr_updated_policy"
    # "main_results/partiprompts/sdxl_dr_ots_high_margin_pseudo_qwen"
    # "main_results/partiprompts/sdxl_dr_ots_high_margin_pseudo_clip"
    # "main_results/partiprompts/sd15_dpo_base_labeled_and_pseudo_unlabeled_clip"
    # "main_results/partiprompts/sd15_dpo_base_labeled_and_pseudo_unlabeled_qwen"
    # "main_results/partiprompts/sd15_dr_ots_clip"
    # "main_results/partiprompts/sd15_dr_ots_qwen"
    "main_results/partiprompts/sd15_dr_ots_high_margin_pseudo_qwen"
)
for result in "${results[@]}"; do
    subfolder=$(echo "$result" | awk -F'/' '{print $3}')
    name="eval_results/main/aescore/${subfolder}"
    echo "Evaluating $name"
    CUDA_VISIBLE_DEVICES=0 python evaluator.py \
        benchmark_type="aescore" \
        image_dir=$result \
        prompt_dir="datasets/eval_prompts/partiprompts.json" \
        name="eval_results/main/aescore/${subfolder}"
done

# CUDA_VISIBLE_DEVICES=4 python evaluator.py \
#     benchmark_type="aescore" \
#     image_dir="main_results/partiprompts/sd15_dpo_25label" \
#     prompt_dir="datasets/eval_prompts/partiprompts.json" \
#     name="eval_results/main/pickscore/sd15_dpo_25label" &

# CUDA_VISIBLE_DEVICES=5 python evaluator.py \
#     benchmark_type="aescore" \
#     image_dir="main_results/partiprompts/sd15_dpo_100label" \
#     prompt_dir="datasets/eval_prompts/partiprompts.json" \
#     name="eval_results/main/pickscore/sd15_dpo_100label" &

