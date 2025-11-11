CUDA_VISIBLE_DEVICES=1 python evaluator.py \
    benchmark_type="hpsv2" \
    image_dir="main_results/hpsv2/sd15_sft_5k_checkpoint-100" \
    prompt_dir="datasets/eval_prompts" \
    name="eval_results/main/hpsv2/sd15_sft_5k_checkpoint-100"  &

# CUDA_VISIBLE_DEVICES=1 python evaluator.py \
#     benchmark_type="hpsv2" \
#     image_dir="main_results/hpsv2/sd15_dr_ots_high_margin_pseudo_clip" \
#     prompt_dir="datasets/eval_prompts" \
#     name="eval_results/main/hpsv2/sd15_dr_ots_high_margin_pseudo_clip_v2"  &

# CUDA_VISIBLE_DEVICES=2 python evaluator.py \
#     benchmark_type="hpsv2" \
#     image_dir="main_results/hpsv2/sdxl_dr_ots_high_margin_pseudo_qwen" \
#     prompt_dir="datasets/eval_prompts" \
#     name="eval_results/main/hpsv2/sdxl_dr_ots_high_margin_pseudo_qwen"  &
