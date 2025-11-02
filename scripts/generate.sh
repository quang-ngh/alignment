CUDA_VISIBLE_DEVICES=0 python generate.py \
    gen_type="hpsv2" \
    model_path="checkpoints/sd15" \
    unet_path="" \
    save_dir="./main_results/hpsv2/sd15_base" \
    version="sd15" \
    noise_path="datasets/hpsv2_noise.pt" \
    guidance_scale=7.5 \
    inference_steps=20 \
    batch_size=20 &

CUDA_VISIBLE_DEVICES=1 python generate.py \
    gen_type="hpsv2" \
    model_path="checkpoints/sd15" \
    save_dir="./main_results/hpsv2/unet_dpo_25label" \
    unet_path="checkpoints/khiem_unet_25label" \
    version="sd15" \
    noise_path="datasets/hpsv2_noise.pt" \
    guidance_scale=7.5 \
    inference_steps=20 \
    batch_size=20 &

CUDA_VISIBLE_DEVICES=2 python generate.py \
    gen_type="hpsv2" \
    model_path="checkpoints/sd15" \
    unet_path="checkpoints/khiem_unet_100label" \
    save_dir="./main_results/hpsv2/unet_dpo_100label" \
    version="sd15" \
    noise_path="datasets/hpsv2_noise.pt" \
    guidance_scale=7.5 \
    inference_steps=20 \
    batch_size=20 &

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

TEST_PROMPTS="datasets/eval_prompts/partiprompts.json"
NOISE_PATH="datasets/partiprompts_noise_sd15.pt"

CUDA_VISIBLE_DEVICES=3 python generate.py \
    gen_type="pickapic_test" \
    model_path="checkpoints/sd15" \
    save_dir="./main_results/partiprompts/sd15_dpo_25label" \
    unet_path="checkpoints/khiem_unet_25label" \
    version="sd15" \
    noise_path=$NOISE_PATH \
    json_path=$TEST_PROMPTS \
    batch_size=20 \
    start_idx=0 \
    guidance_scale=7.5 \
    inference_steps=20 \
    end_idx=-1 &


CUDA_VISIBLE_DEVICES=4 python generate.py \
    gen_type="pickapic_test" \
    model_path="checkpoints/sd15" \
    save_dir="./main_results/partiprompts/sd15_dpo_100label" \
    unet_path="checkpoints/khiem_unet_100label" \
    version="sd15" \
    noise_path=$NOISE_PATH \
    json_path=$TEST_PROMPTS \
    batch_size=20 \
    start_idx=0 \
    guidance_scale=7.5 \
    inference_steps=20 \
    end_idx=-1 &


CUDA_VISIBLE_DEVICES=5 python generate.py \
    gen_type="pickapic_test" \
    model_path="checkpoints/sd15" \
    save_dir="./main_results/partiprompts/sd15_base" \
    unet_path="" \
    version="sd15" \
    noise_path=$NOISE_PATH \
    json_path=$TEST_PROMPTS \
    batch_size=20 \
    start_idx=0 \
    guidance_scale=7.5 \
    inference_steps=20 \
    end_idx=-1 &


# CUDA_VISIBLE_DEVICES=4 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir=$OUTPUT_DIR \
#     unet_path=$DPO_FIFA_UNET \
#     version="sd15" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=20 \
#     start_idx=240 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=320 &


# CUDA_VISIBLE_DEVICES=5 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir=$OUTPUT_DIR \
#     unet_path=$DPO_FIFA_UNET \
#     version="sd15" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=20 \
#     start_idx=320 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &




