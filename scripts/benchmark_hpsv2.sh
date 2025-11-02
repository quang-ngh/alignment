CUDA_VISIBLE_DEVICES=5 python evaluator.py \
    benchmark_type="hpsv2" \
    image_dir="main_results/hpsv2/sdxl_base" \
    prompt_dir="datasets/eval_prompts" \
    name="eval_results/main/hpsv2/sdxl_base" &

# CUDA_VISIBLE_DEVICES=1 python evaluator.py \
#     benchmark_type="hpsv2" \
#     image_dir="main_results/hpsv2/unet_dpo_25label" \
#     prompt_dir="datasets/eval_prompts" \
#     name="eval_results/main/hpsv2/unet_dpo_25label" &

# CUDA_VISIBLE_DEVICES=2 python evaluator.py \
#     benchmark_type="hpsv2" \
#     image_dir="main_results/hpsv2/unet_dpo_100label" \
#     prompt_dir="datasets/eval_prompts" \
#     name="eval_results/main/hpsv2/unet_dpo_100label" &

