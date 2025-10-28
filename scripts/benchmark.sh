CUDA_VISIBLE_DEVICES=1 python evaluator.py \
    benchmark_type="pickscore" \
    image_dir="output/khiem_unet" \
    prompt_dir="datasets/eval_prompts/pickapic_test_prompts.json" \
    name="pickscore_khiem_unet_pickapic_test" \
