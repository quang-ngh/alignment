CUDA_VISIBLE_DEVICES=0 python generate_pseudo_label.py \
    --input_json "datasets/manifest_5k_animated/5k_25_75/labeled.json" \
    --output_json "datasets/manifest_5k_animated/5k_25_75/pseudo_labeled_qwen.json" \
    --image_base_dir "datasets/pickapic-5k-animated-flipped-30pct/data/train" \
    --strategy "qwen_vlm" \
    --model_name "./checkpoints/qwen_7b" \
    --batch_size 1 &

CUDA_VISIBLE_DEVICES=1 python generate_pseudo_label.py \
    --input_json "datasets/manifest_5k_animated/5k_25_75/unlabeled.json" \
    --output_json "datasets/manifest_5k_animated/5k_25_75/pseudo_unlabeled_qwen.json" \
    --image_base_dir "datasets/pickapic-5k-animated-flipped-30pct/data/train" \
    --strategy "qwen_vlm" \
    --model_name "./checkpoints/qwen_7b" \
    --batch_size 1 &
    
# CUDA_VISIBLE_DEVICES=2 python generate_pseudo_label.py \
#     --input_json "datasets/manifest_high_margin/from_20k_high_margin_25_75/labeled.json" \
#     --output_json "datasets/manifest_high_margin/from_20k_high_margin_25_75/pseudo_labeled_qwen.json" \
#     --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
#     --strategy "qwen_vlm" \
#     --model_name "./checkpoints/qwen_7b" \
#     --batch_size 1 &

# CUDA_VISIBLE_DEVICES=3 python generate_pseudo_label.py \
#     --input_json "datasets/manifest_high_margin/from_20k_high_margin_25_75/unlabeled.json" \
#     --output_json "datasets/manifest_high_margin/from_20k_high_margin_25_75/pseudo_unlabeled_qwen.json" \
#     --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
#     --strategy "qwen_vlm" \
#     --model_name "./checkpoints/qwen_7b" \
#     --batch_size 1 &
    
# CUDA_VISIBLE_DEVICES=4 python generate_pseudo_label.py \
#     --input_json "datasets/manifest_high_margin/from_50k_high_margin_25_75/labeled.json" \
#     --output_json "datasets/manifest_high_margin/from_50k_high_margin_25_75/pseudo_labeled_qwen.json" \
#     --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
#     --strategy "qwen_vlm" \
#     --model_name "./checkpoints/qwen_7b" \
#     --batch_size 1 &

# CUDA_VISIBLE_DEVICES=5 python generate_pseudo_label.py \
#     --input_json "datasets/manifest_high_margin/from_50k_high_margin_25_75/unlabeled.json" \
#     --output_json "datasets/manifest_high_margin/from_50k_high_margin_25_75/pseudo_unlabeled_qwen.json" \
#     --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
#     --strategy "qwen_vlm" \
#     --model_name "./checkpoints/qwen_7b" \
#     --batch_size 1 &
    
# CUDA_VISIBLE_DEVICES=6 python generate_pseudo_label.py \
#     --input_json "datasets/manifest_high_margin/from_100k_high_margin_25_75/labeled.json" \
#     --output_json "datasets/manifest_high_margin/from_100k_high_margin_25_75/pseudo_labeled_qwen.json" \
#     --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
#     --strategy "qwen_vlm" \
#     --model_name "./checkpoints/qwen_7b" \
#     --batch_size 1 &

# CUDA_VISIBLE_DEVICES=7 python generate_pseudo_label.py \
#     --input_json "datasets/manifest_high_margin/from_100k_high_margin_25_75/unlabeled.json" \
#     --output_json "datasets/manifest_high_margin/from_100k_high_margin_25_75/pseudo_unlabeled_qwen.json" \
#     --image_base_dir "datasets/FiFA-100k-sorted/data/train" \
#     --strategy "qwen_vlm" \
#     --model_name "./checkpoints/qwen_7b" \
#     --batch_size 1 &
wait 