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
    # "main_results/partiprompts/sd15_ablate2_drdpo_10k_pseudo"
    # "main_results/partiprompts/ssh d15_dr_sota_test_5k_label_5k_unlabel"
    # "main_results/partiprompts/sd15_ablate2_drdpo_50k_pseudo"
    # "main_results/partiprompts/sd15_ablate2_drdpo_100k_pseudo"
    # "main_results/partiprompts/sd15_dr_sota_test_5k_label_5k_unlabel_ckpt400"
    # "main_results/partiprompts/sd15_ablate2_dpo_10k_pseudo"
    # "main_results/partiprompts/sd15_ablate2_dpo_50k_pseudo"
    # "main_results/partiprompts/sd15_ablate2_dpo_100k_pseudo"
    # "main_results/partiprompts/sd15_dr_sota_test_5k_label_5k_unlabel_checkpoint-200"
    # "main_results/partiprompts/sd15_dr_ots_1k2_8k8_qwen_checkpoint-200"
    # "main_results/partiprompts/sd15_dr_ots_1k2_8k8_qwen_checkpoint-300"
    # "main_results/partiprompts/sd15_dr_ots_1k2_8k8_qwen_checkpoint-400"
    # "main_results/partiprompts/sd15_dr_ots_1k2_38k8_qwen_checkpoint-200"
    # "main_results/partiprompts/sd15_dr_ots_1k2_38k8_qwen_checkpoint-300"
    # "main_results/partiprompts/sd15_dr_ots_1k2_38k8_qwen_checkpoint-400"
    # "main_results/partiprompts/sd15_dr_ots_1k2_98k8_qwen_checkpoint-200"
    # "main_results/partiprompts/sd15_dr_ots_1k2_98k8_qwen_checkpoint-300"
    # "main_results/partiprompts/sd15_dr_ots_1k2_98k8_qwen_checkpoint-400"
    # "main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-200"
    # "main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-300"
    # "main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-400"
    "main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-500"
    "main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-600"
    "main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-700"
)

for result in "${results[@]}"; do
    subfolder=$(echo "$result" | awk -F'/' '{print $3}')
    name="eval_results/main/pickscore/${subfolder}"
    echo "Evaluating $name"
    CUDA_VISIBLE_DEVICES=1 python evaluator.py \
        benchmark_type="pickscore" \
        image_dir=$result \
        prompt_dir="datasets/eval_prompts/partiprompts.json" \
        name="eval_results/main/pickscore/sd15_ablate_dpo_${subfolder}"  &
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

