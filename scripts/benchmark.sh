CUDA_VISIBLE_DEVICES=1 python evaluator.py \
    benchmark_type="pickscore" \
    image_dir="output/sd15_dpo_fifa_ckpt400" \
    prompt_dir="datasets/eval_prompts/pickapic_test_prompts.json" \
    name="pickscore_sd15_dpo_fifa_ckpt400_pickapic_test" \
