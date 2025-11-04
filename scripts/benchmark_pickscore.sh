CUDA_VISIBLE_DEVICES=6 python evaluator.py \
    benchmark_type="pickscore" \
    image_dir="main_results/partiprompts/sd15_dr_ots_high_margin_pseudo_qwen" \
    prompt_dir="datasets/eval_prompts/partiprompts.json" \
    name="eval_results/main/pickscore/sd15_dr_ots_high_margin_pseudo_qwen" &

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

