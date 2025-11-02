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




