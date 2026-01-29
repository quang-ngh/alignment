CKPT=/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b

# CUDA_VISIBLE_DEVICES=0 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$CKPT \
#     unet_path="training_runs/sdxl_sft_5k_high_margin/checkpoint-100/unet" \
#     save_dir="./main_results/hpsv2/sdxl_sft_5k_high_margin_checkpoint-100" \
#     version="sdxl" \
#     noise_path="datasets/hpsv2_noise_xl.pt" \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     batch_size=8 &

# CUDA_VISIBLE_DEVICES=4 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$CKPT \
#     save_dir="./main_results/hpsv2/sdxl_dr_ots_high_margin_pseudo_clip" \
#     unet_path="training_runs/sdxl_dr_ots_high_margin_pseudo_clip/checkpoint-50/unet" \
#     version="sdxl" \
#     noise_path="datasets/hpsv2_noise_xl.pt" \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     batch_size=8 &

# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$CKPT \
#     unet_path="training_runs/sdxl_dr_ots_high_margin_pseudo_qwen/checkpoint-50/unet" \
#     save_dir="./main_results/hpsv2/sdxl_dr_ots_high_margin_pseudo_qwen" \
#     version="sdxl" \
#     noise_path="datasets/hpsv2_noise_xl.pt" \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     batch_size=8 &

# CUDA_VISIBLE_DEVICES=1 python generate.py \
#     gen_type="pickapic_test" \
#     model_path="checkpoints/sd15" \
#     save_dir="./main_results/partiprompts/sdxl_sft_5k_high_margin_checkpoint-50" \
#     unet_path="training_runs/sdxl_sft_5k_high_margin/checkpoint-50/unet" \
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
TEST_PROMPTS="datasets/eval_prompts/partiprompts.json"
# NOISE_PATH="datasets/partiprompts_noise_sdxl.pt"

# CUDA_VISIBLE_DEVICES=0 python generate.py \
#     gen_type="pickapic_test" \
#     model_path=$CKPT \
#     save_dir="main_results/partiprompts/sdxl_sft_5k_high_margin_checkpoint-100" \
#     unet_path="checkpoints/khiem_sdxl" \
#     version="sdxl" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=8 \
#     start_idx=0 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=800 &


# CUDA_VISIBLE_DEVICES=1 python generate.py \
#     gen_type="pickapic_test" \
#     model_path=$CKPT \
#     save_dir="main_results/partiprompts/sdxl_sft_5k_high_margin_checkpoint-100" \
#     unet_path="training_runs/sdxl_sft_5k_high_margin/checkpoint-100/unet" \
#     version="sdxl" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=8 \
#     start_idx=800 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &


# CUDA_VISIBLE_DEVICES=2 python generate.py \
#     gen_type="pickapic_test" \
#     model_path=$CKPT \
#     save_dir=$OUTPUT_DIR \
#     unet_path=$UNET \
#     version="sdxl" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=8 \
#     start_idx=800 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=1200 &


# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="pickapic_test" \
#     model_path=$CKPT \
#     save_dir=$OUTPUT_DIR \
#     unet_path=$UNET \
#     version="sdxl" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=8 \
#     start_idx=1200 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &

# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="pickapic_test" \
#     model_path=$CKPT \
#     save_dir=$OUTPUT_DIR \
#     unet_path=$DPO_FIFA_UNET \
#     version="sdxl" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=8 \
#     start_idx=1200 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &


# CUDA_VISIBLE_DEVICES=5 python generate.py \
#     gen_type="pickapic_test" \
#     model_path=$CKPT \
#     save_dir="./main_results/partiprompts/sdxl_base" \
#     unet_path="" \
#     version="sdxl" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=8 \
#     start_idx=800 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=1000 &


# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="pickapic_test" \
#     model_path=$CKPT \
#     save_dir="./main_results/partiprompts/sdxl_base" \
#     unet_path="" \
#     version="sdxl" \
#     noise_path=$NOISE_PATH \
#     json_path=$TEST_PROMPTS \
#     batch_size=8 \
#     start_idx=1000 \
#     guidance_scale=7.5 \
#     inference_steps=20 \
#     end_idx=-1 &


############# HPSv2

# UNET_DPO_25="training_runs/sdxl_dpo_fifa5k_high_margin_25/checkpoint-50/unet"
# UNET_DPO_100="training_runs/sdxl_dpo_fifa5k_high_margin_100_ddp/checkpoint-50/unet"

# SAVE_DIR_25="./main_results/hpsv2/sdxl_dpo_fifa5k_high_margin_25"
# SAVE_DIR_100="./main_results/hpsv2/sdxl_dpo_fifa5k_high_margin_100"
# NOISE_PATH="datasets/hpsv2_noise_xl.pt"

# CUDA_VISIBLE_DEVICES=6 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$CKPT \
#     unet_path=$UNET_DPO_25 \
#     version="sdxl" \
#     save_dir=$SAVE_DIR_25 \
#     noise_path=$NOISE_PATH \
#     batch_size=8 \
#     inference_steps=20 \
#     guidance_scale=7.5 &

# CUDA_VISIBLE_DEVICES=7 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$CKPT \
#     unet_path=$UNET_DPO_100 \
#     version="sdxl" \
#     save_dir=$SAVE_DIR_100 \
#     noise_path=$NOISE_PATH \
#     batch_size=8 \
#     inference_steps=20 \
#     guidance_scale=7.5

# CUDA_VISIBLE_DEVICES=3 python generate.py \
#     gen_type="hpsv2" \
#     model_path=$CKPT \
#     unet_path="training_runs/sdxl_dr_ots_high_margin_pseudo_qwen/checkpoint-50/unet" \
#     version="sdxl" \
#     save_dir="./main_results/hpsv2/sdxl_dr_ots_high_margin_pseudo_qwen" \
#     noise_path=$NOISE_PATH \
#     batch_size=8 \
#     inference_steps=20 \
#     guidance_scale=7.5


MODELS=(
    # "training_runs/sdxl_dpo_fifa_25"
    # "training_runs/sdxl_dpo_fifa_100"
    "training_runs/sdxl_dr_dpo_pseudo"
    # "training_runs/sdxl_dpo_pseudo/checkpoint-50/unet"
    # "training_runs/sdxl_dpo_pseudo"
)

SEEDS=(
    # 28548 
    # 43070 
    # 63817 
    # 95457 
    # 62178 
    # 76948 
    # 96160 
    # 51399 
    # 9443 
    # 48870 
    29961 
    64793 
    20817 
    24652 
    33449
)
for model in "${MODELS[@]}"; do
    model_name=$(echo "$model" | awk -F'/' '{print $2}')
    save_dir="main_supp/partiprompts/${model_name}"
    noise_path=""
    for run in {1..5}
    do
        echo "Run $run/15"
        seed=${SEEDS[$run-1]}
        echo "Using seed $seed"
        # CUDA_VISIBLE_DEVICES=2 python generate.py \
        #     random_seed=$seed \
        #     gen_type="pickapic_test" \
        #     model_path=$CKPT \
        #     save_dir=${save_dir} \
        #     unet_path=$model \
        #     version="sdxl" \
        #     noise_path="" \
        #     json_path=$TEST_PROMPTS \
        #     batch_size=8 \
        #     start_idx=0 \
        #     guidance_scale=7.5 \
        #     inference_steps=20 \
        #     end_idx=200 &

        # CUDA_VISIBLE_DEVICES=3 python generate.py \
        #     random_seed=$seed \
        #     gen_type="pickapic_test" \
        #     model_path=$CKPT \
        #     save_dir=${save_dir} \
        #     unet_path=$model \
        #     version="sdxl" \
        #     noise_path="" \
        #     json_path=$TEST_PROMPTS \
        #     batch_size=8 \
        #     start_idx=200 \
        #     guidance_scale=7.5 \
        #     inference_steps=20 \
        #     end_idx=400 &

        # CUDA_VISIBLE_DEVICES=2 python generate.py \
        #     random_seed=$seed \
        #     gen_type="pickapic_test" \
        #     model_path=$CKPT \
        #     save_dir=${save_dir} \
        #     unet_path=$model \
        #     version="sdxl" \
        #     noise_path="" \
        #     json_path=$TEST_PROMPTS \
        #     batch_size=4 \
        #     start_idx=400 \
        #     guidance_scale=7.5 \
        #     inference_steps=20 \
        #     end_idx=600 &

        # CUDA_VISIBLE_DEVICES=3 python generate.py \
        #     random_seed=$seed \
        #     gen_type="pickapic_test" \
        #     model_path=$CKPT \
        #     save_dir=${save_dir} \
        #     unet_path=$model \
        #     version="sdxl" \
        #     noise_path="" \
        #     json_path=$TEST_PROMPTS \
        #     batch_size=4 \
        #     start_idx=600 \
        #     guidance_scale=7.5 \
        #     inference_steps=20 \
        #     end_idx=800 &

        CUDA_VISIBLE_DEVICES=0 python generate.py \
            random_seed=$seed \
            gen_type="pickapic_test" \
            model_path=$CKPT \
            save_dir=${save_dir} \
            unet_path=$model \
            version="sdxl" \
            noise_path="" \
            json_path=$TEST_PROMPTS \
            batch_size=4 \
            start_idx=800 \
            guidance_scale=7.5 \
            inference_steps=20 \
            end_idx=1000 &

        CUDA_VISIBLE_DEVICES=1 python generate.py \
            random_seed=$seed \
            gen_type="pickapic_test" \
            model_path=$CKPT \
            save_dir=${save_dir} \
            unet_path=$model \
            version="sdxl" \
            noise_path="" \
            json_path=$TEST_PROMPTS \
            batch_size=4 \
            start_idx=1000 \
            guidance_scale=7.5 \
            inference_steps=20 \
            end_idx=1200 &

        # CUDA_VISIBLE_DEVICES=6 python generate.py \
        #     random_seed=$seed \
        #     gen_type="pickapic_test" \
        #     model_path=$CKPT \
        #     save_dir=${save_dir} \
        #     unet_path=$model \
        #     version="sdxl" \
        #     noise_path="" \
        #     json_path=$TEST_PROMPTS \
        #     batch_size=4 \
        #     start_idx=1200 \
        #     guidance_scale=7.5 \
        #     inference_steps=20 \
        #     end_idx=-1 &
        wait
    done
done
