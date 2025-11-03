CUDA_VISIBLE_DEVICES=6 python generate_pseudo_label.py \
    --input_json "datasets/manifest_high_margin/from_5k_high_margin_25_75/labeled.json" \
    --output_json "datasets/manifest_high_margin/from_5k_high_margin_25_75/pseudo_labeled_clip.json" \
    --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
    --strategy "clipscore" \
    --model_name "openai/clip-vit-base-patch32" \
    --batch_size 1 \
    &

CUDA_VISIBLE_DEVICES=7 python generate_pseudo_label.py \
    --input_json "datasets/manifest_high_margin/from_5k_high_margin_25_75/labeled.json" \
    --output_json "datasets/manifest_high_margin/from_5k_high_margin_25_75/pseudo_labeled_qwen.json" \
    --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
    --strategy "qwen_vlm" \
    --model_name "./checkpoints/qwen_7b" \
    --batch_size 1 \
    