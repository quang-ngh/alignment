# UNET="/common/users/hn315/alignment/sd15_dr_ots_high_margin_pseudo_qwen_20k/checkpoint-100/unet"
# UNET_50="training_runs/sd15_dr_ots_high_margin_pseudo_qwen_50k/checkpoint-100/unet"
# SAVE_DIR="./main_results/partiprompts/sd15_dr_ots_high_margin_pseudo_qwen_20k"
# SAVE_DIR_50="./main_results/partiprompts/sd15_dr_ots_high_margin_pseudo_qwen_50k"

# UNET_ABLATE_100="training_runs/ablate_100k/checkpoint-100/unet"
# SAVE_DIR="./main_results/partiprompts/sd15_dr_ots_high_margin_pseudo_ablate_100k"

# UNET_DRDPO_AB10k="training_runs/sd15_dr_ots_1k2_98k8_qwen/checkpoint-100/unet"
# SAVE_DIR="./main_results/partiprompts/sd15_ablate2_drdpo_100k_pseudo"

# UNET="training_runs/sd15_dr_ots_hpsv2_5k_last_qwen/checkpoint-100/unet"
# UNET_DRDPO_KHIEM=""


CUDA_VISIBLE_DEVICES=6 python generate.py \
    gen_type="pickapic_test" \
    model_path="checkpoints/sd15" \
    unet_path="training_runs/sd15_dpo_flip80_pseudo/checkpoint-100/unet" \
    save_dir="./main_results/partiprompts/sd15_dpo_flip80_pseudo_ckpt100" \
    version="sd15" \
    noise_path="datasets/partiprompts_noise_sd15.pt" \
    json_path="datasets/eval_prompts/partiprompts.json" \
    batch_size=8 \
    start_idx=0 \
    guidance_scale=7.5 \
    inference_steps=20 \
    end_idx=800 &

CUDA_VISIBLE_DEVICES=7 python generate.py \
    gen_type="pickapic_test" \
    model_path="checkpoints/sd15" \
    unet_path="training_runs/sd15_dpo_flip80_pseudo/checkpoint-100/unet" \
    save_dir="./main_results/partiprompts/sd15_dpo_flip80_pseudo_ckpt100" \
    version="sd15" \
    noise_path="datasets/partiprompts_noise_sd15.pt" \
    json_path="datasets/eval_prompts/partiprompts.json" \
    batch_size=8 \
    start_idx=800 \
    guidance_scale=7.5 \
    inference_steps=20 \
    end_idx=-1 &

# CUDA_VISIBLE_DEVICES=2 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path=$UNET_DRDPO_KHIEM \
#     save_dir="./main_results/partiprompts/sd15_drdpo_25_75_khiem" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=8 \
#     start_idx=800 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=1200 &

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path=$UNET_DRDPO_KHIEM \
#     save_dir="./main_results/partiprompts/sd15_drdpo_25_75_khiem" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=8 \
#     start_idx=1200 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="training_runs/sd15_test_sota_5k_label_5k_unlabel_v2/checkpoint-700/unet" \
#     save_dir="./main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-700" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=8 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=800 &

# CUDA_VISIBLE_DEVICES=7 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="training_runs/sd15_test_sota_5k_label_5k_unlabel_v2/checkpoint-700/unet" \
#     save_dir="./main_results/partiprompts/sd15_test_sota_5k_label_5k_unlabel_v2_checkpoint-700" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=800 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &


# CUDA_VISIBLE_DEVICES=2 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path=${UNET_DRDPO_AB10k} \
#     save_dir=${SAVE_DIR} \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=800 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=1200 &

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path=${UNET_DRDPO_AB10k} \
#     save_dir=${SAVE_DIR} \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=1200 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=2 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="training_runs/ablate_dpo_50k_pseudo/checkpoint-100/unet" \
#     save_dir="./main_results/partiprompts/sd15_ablate_dpo_50k_pseudo" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="training_runs/ablate_dpo_100k_pseudo/checkpoint-100/unet" \
#     save_dir="./main_results/partiprompts/sd15_ablate_dpo_100k_pseudo" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=5 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path=$UNET \
#     save_dir=$SAVE_DIR \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=500 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=1000 &

# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path=$UNET \
#     save_dir=$SAVE_DIR \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=1000 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path=$UNET \
#     save_dir=$SAVE_DIR \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=20 \
#     start_idx=1200 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=1 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./output/base_sd15_pickapic_test_default_gs" \
#     unet_path="" \
#     version="sd15" \
#     noise_path="datasets/pickapic_test_noise.pt" \
#     json_path="datasets/eval_prompts/pickapic_test_prompts.json" \
#     batch_size=20 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     end_idx=80 &


# CUDA_VISIBLE_DEVICES=2 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./output/base_sd15_pickapic_test_default_gs" \
#     unet_path="" \
#     version="sd15" \
#     noise_path="datasets/pickapic_test_noise.pt" \
#     json_path="datasets/eval_prompts/pickapic_test_prompts.json" \
#     batch_size=20 \
#     start_idx=80 \
#     guidance_scale=7.5 \
#     end_idx=160 &


# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./output/base_sd15_pickapic_test_default_gs" \
#     unet_path="" \
#     version="sd15" \
#     noise_path="datasets/pickapic_test_noise.pt" \
#     json_path="datasets/eval_prompts/pickapic_test_prompts.json" \
#     batch_size=20 \
#     start_idx=160 \
#     guidance_scale=7.5 \
#     end_idx=240 &


# CUDA_VISIBLE_DEVICES=4 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./output/base_sd15_pickapic_test_default_gs" \
#     unet_path="" \
#     version="sd15" \
#     noise_path="datasets/pickapic_test_noise.pt" \
#     json_path="datasets/eval_prompts/pickapic_test_prompts.json" \
#     batch_size=20 \
#     start_idx=240 \
#     guidance_scale=7.5 \
#     end_idx=320 &

# CUDA_VISIBLE_DEVICES=5 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./output/base_sd15_pickapic_test_default_gs" \
#     unet_path="" \
#     version="sd15" \
#     noise_path="datasets/pickapic_test_noise.pt" \
#     json_path="datasets/eval_prompts/pickapic_test_prompts.json" \
#     batch_size=20 \
#     start_idx=320 \
#     guidance_scale=7.5 \
#     end_idx=400 &

# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./output/base_sd15_pickapic_test_default_gs" \
#     unet_path="" \
#     version="sd15" \
#     noise_path="datasets/pickapic_test_noise.pt" \
#     json_path="datasets/eval_prompts/pickapic_test_prompts.json" \
#     batch_size=20 \
#     start_idx=400 \
#     guidance_scale=7.5 \
#     end_idx=-1 &


#########
# DPO_FIFA_UNET="checkpoints/khiem_unet_25label"
# OUTPUT_DIR="./main_results/partiprompts/sd15_dpo_25label"

# TEST_PROMPTS="datasets/eval_prompts/partiprompts.json"
# NOISE_PATH="datasets/partiprompts_noise_sd15.pt"

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./main_results/partiprompts/sd15_dpo_25label" \
#     unet_path="checkpoints/khiem_unet_25label" \
#     version="sd15" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=20 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &


# CUDA_VISIBLE_DEVICES=4 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./main_results/partiprompts/sd15_dpo_100label" \
#     unet_path="checkpoints/khiem_unet_100label" \
#     version="sd15" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=20 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &


# CUDA_VISIBLE_DEVICES=5 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./main_results/partiprompts/sd15_base" \
#     unet_path="" \
#     version="sd15" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=20 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &


# # CUDA_VISIBLE_DEVICES=4 python generate.py \
# #     gen_type="pickapic_test" \
# #     model_path="checkpoints/sd15" \
# #     save_dir=$OUTPUT_DIR \
# #     unet_path=$DPO_FIFA_UNET \
# #     version="sd15" \
# #     noise_path=$NOISE_PATH \
# #     json_path=$TEST_PROMPTS \
# #     batch_size=20 \
# #     start_idx=240 \
# #     guidance_scale=7.5 \
# #     inference_steps=20 \
# #     end_idx=320 &


# # CUDA_VISIBLE_DEVICES=5 python generate.py \
# #     gen_type="pickapic_test" \
# #     model_path="checkpoints/sd15" \
# #     save_dir=$OUTPUT_DIR \
# #     unet_path=$DPO_FIFA_UNET \
# #     version="sd15" \
# #     noise_path=$NOISE_PATH \
# #     json_path=$TEST_PROMPTS \
# #     batch_size=20 \
# #     start_idx=320 \
# #     guidance_scale=7.5 \
# #     inference_steps=20 \
# #     end_idx=-1 &



############# HPSv2
# MODELS="checkpoints/sd15"
# UNET_DR_DPO_QWEN="training_runs/sd15_dr_ots_high_margin_pseudo_qwen/checkpoint-100/unet"
# UNET_DR_DPO_CLIP="training_runs/sd15_dr_ots_high_margin_pseudo_clip/checkpoint-200/unet"
# UNET_ABLATE_100="training_runs/ablate_100k/checkpoint-100/unet"
# SAVE_DIR_ABLATE_100="./main_results/hpsv2/sd15_dr_ots_high_margin_pseudo_ablate_100k"

# UNET_DPO_QWEN="training_runs/sd15_dpo_base_labeled_and_pseudo_unlabeled_qwen/checkpoint-100/unet"
# UNET_DPO_CLIP="training_runs/sd15_dpo_base_labeled_and_pseudo_unlabeled_clip/checkpoint-100/unet"
# SAVE_DIR_QWEN="./main_results/hpsv2/sd15_dr_ots_high_margin_pseudo_qwen"
# SAVE_DIR_CLIP="./main_results/hpsv2/sd15_dr_ots_high_margin_pseudo_clip"
# SAVE_DIR_ABLATE_100="./main_results/hpsv2/sd15_dr_ots_high_margin_pseudo_ablate_100"


# NOISE_PATH="datasets/hpsv2_noise.pt"

# CUDA_VISIBLE_DEVICES=7 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$MODELS \
#     unet_path="training_runs/sd15_dpo_flip80_pseudo/checkpoint-100/unet" \
#     version="sd15" \
#     save_dir="./main_results/hpsv2/sd15_dpo_flip80_pseudo_ckpt100" \
#     noise_path=$NOISE_PATH \
#     batch_size=20 \
#     inference_steps=20 \
#     guidance_scale=7.5 &

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$MODELS \
#     unet_path="training_runs/sd15_dr_dpo_pseudo_smolvlm/checkpoint-100/unet" \
#     version="sd15" \
#     save_dir="./main_results/hpsv2/sd15_dr_dpo_pseudo_smolvlm_ckpt100" \
#     noise_path=$NOISE_PATH \
#     batch_size=20 \
#     inference_steps=20 \
#     guidance_scale=7.5 &

# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$MODELS \
#     unet_path="training_runs/sd15_dr_dpo_98k_unl/checkpoint-100/unet" \
#     version="sd15" \
#     save_dir="./main_results/hpsv2/sd15_dr_dpo_98k_unl" \
#     noise_path=$NOISE_PATH \
#     batch_size=20 \
#     inference_steps=20 \
#     guidance_scale=7.5 &

# CUDA_VISIBLE_DEVICES=7 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$MODELS \
#     unet_path="training_runs/sd15_dpo_8k_unl" \
#     version="sd15" \
#     save_dir="./main_results/hpsv2/sd15_dpo_8k_unl" \
#     noise_path=$NOISE_PATH \
#     batch_size=20 \
#     inference_steps=20 \
#     guidance_scale=7.5

# models=(
#     # "training_runs/ablate_dpo_10k_pseudo/checkpoint-100/unet"
#     # "training_runs/ablate_dpo_20k_pseudo/checkpoint-100/unet"
#     # "training_runs/ablate_dpo_50k_pseudo/checkpoint-100/unet"
#     # "training_runs/ablate_dpo_100k_pseudo/checkpoint-100/unet"

# )

# for unet_path in "${models[@]}"; do
#     subfolder=$(echo "$unet_path" | awk -F'/' '{print $2}')
#     save_dir="./main_results/partiprompts/sd15_ablate_dpo_${subfolder}"

#     echo "Generating $save_dir"

#     CUDA_VISIBLE_DEVICES=2 python generate.py \
#         gen_type="pickapic_test" \
#         model_path="checkpoints/sd15" \
#         unet_path=$unet_path \
#         save_dir=$save_dir \
#         version="sd15" \
#         noise_path="datasets/partiprompts_noise_sd15.pt" \
#         json_path="datasets/eval_prompts/partiprompts.json" \
#         batch_size=20 \
#         start_idx=0 \
#         guidance_scale=7.5 \
#         inference_steps=20 \
#         end_idx=-1 &

#     CUDA_VISIBLE_DEVICES=3 python generate.py \
#         gen_type="pickapic_test" \
#         model_path="checkpoints/sd15" \
#         unet_path=$unet_path \
#         save_dir=$save_dir \
#         version="sd15" \
#         noise_path="datasets/partiprompts_noise_sd15.pt" \
#         json_path="datasets/eval_prompts/partiprompts.json" \
#         batch_size=20 \
#         start_idx=0 \
#         guidance_scale=7.5 \
#         inference_steps=20 \
#         end_idx=-1 &

#     CUDA_VISIBLE_DEVICES=4 python generate.py \
#         gen_type="pickapic_test" \
#         model_path="checkpoints/sd15" \
#         unet_path=$unet_path \
#         save_dir=$save_dir \
#         version="sd15" \
#         noise_path="datasets/partiprompts_noise_sd15.pt" \
#         json_path="datasets/eval_prompts/partiprompts.json" \
#         batch_size=20 \
#         start_idx=0 \
#         guidance_scale=7.5 \
#         inference_steps=20 \
#         end_idx=-1 &

#     CUDA_VISIBLE_DEVICES=7 python generate.py \
#         gen_type="pickapic_test" \
#         model_path="checkpoints/sd15" \
#         unet_path=$unet_path \
#         save_dir=$save_dir \
#         version="sd15" \
#         noise_path="datasets/partiprompts_noise_sd15.pt" \
#         json_path="datasets/eval_prompts/partiprompts.json" \
#         batch_size=20 \
#         start_idx=0 \
#         guidance_scale=7.5 \
#         inference_steps=20 \
#         end_idx=-1 &
# done 
# fi


# CUDA_VISIBLE_DEVICES=0 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="checkpoints/sd15_dpo_hpdv2_1k25" \
#     save_dir="./main_results/partiprompts/sd15_dpo_hpdv2_1k25_khiem" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=8 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=1 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="checkpoints/sd15_dpo_hpdv2_5k" \
#     save_dir="./main_results/partiprompts/sd15_dpo_hpdv2_5k_khiem" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=8 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=2 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="checkpoints/sd15_dpo_pseudo_hpdv2" \
#     save_dir="./main_results/partiprompts/sd15_dpo_pseudo_hpdv2_khiem" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=8 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     unet_path="checkpoints/sd15_dr_pseudo_hpdv2" \
#     save_dir="./main_results/partiprompts/sd15_dr_pseudo_hpdv2_khiem" \
#     version="sd15" \
#     noise_path="datasets/partiprompts_noise_sd15.pt" \
#     json_path="datasets/eval_prompts/partiprompts.json" \
#     batch_size=8 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &
wait