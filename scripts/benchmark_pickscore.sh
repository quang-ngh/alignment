# results=(
#     # "main_results/partiprompts/sd15_ablate_dpo_10k_pseudo"
#     # "main_results/partiprompts/sd15_ablate_dpo_20k_pseudo"
#     # "main_results/partiprompts/sd15_ablate_dpo_50k_pseudo"
#     # "main_results/partiprompts/sd15_ablate_dpo_100k_pseudo"
#     # "main_results/partiprompts/sdxl_dr_updated_policy"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-200"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-300"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-400"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-500"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-600"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-700"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-800"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-900"
#     # "main_results/partiprompts/sdxl_ablate_drdpo_20k_checkpoint-1000"
#     # "main_results/partiprompts/sdxl_sft_5k_high_margin_checkpoint-50"
#     # "main_results/partiprompts/sd15_sft_5k_checkpoint-100"
#     # "./main_results/partiprompts/sdxl_sft_5k_high_margin_checkpoint-100"
#     # "main_results/partiprompts/sdxl_sft_5k_high_margin_checkpoint-100"
#     # "main_results/partiprompts/sd15_dr_ots_hpsv2_5k_last_qwen"
#     # "main_results/partiprompts/sd15_dr_ots_hpsv2_5k_last_qwen_ckpt100"
#     # "main_results/partiprompts/sd15_dr_dpo_pseudo_flip_ckpt100/random_seed_999"
#     # "main_results/partiprompts/sd15_dpo_smolvlm_pseudo_ckpt100/random_seed_999"
#     # "main_results/partiprompts/sd15_dr_dpo_pseudo_smolvlm_ckpt100/random_seed_999"
#     # "main_results/partiprompts/sd15_dr_dpo_pseudo_flip80_ckpt100/random_seed_999"
#     # "main_results/partiprompts/sd15_dpo_flip80_pseudo_ckpt100/random_seed_999"
#     # "main_results/rebuttal/noise_flip/10_percent/random_seed_999"
#     # "main_results/rebuttal/noise_flip/20_percent/random_seed_999"
#     # "main_results/rebuttal/noise_flip/30_percent/random_seed_999"
#     # "main_results/rebuttal/noise_flip/40_percent/random_seed_999"
#     # "main_results/rebuttal/noise_flip/50_percent/random_seed_999"
#     "main_results/rebuttal/ablate_ratio/50_50/random_seed_999"
#     "main_results/rebuttal/ablate_ratio/75_25/random_seed_999"
#     "main_results/rebuttal/increase_labeled_size/100_4900/random_seed_999"
#     "main_results/rebuttal/increase_labeled_size/200_4800/random_seed_999"
#     "main_results/rebuttal/increase_labeled_size/500_4500/random_seed_999"
#     "main_results/rebuttal/small_synthetic_portion/0.1_percent/random_seed_999"
#     "main_results/rebuttal/small_synthetic_portion/0.05_percent/random_seed_999"
# )

# for result in "${results[@]}"; do
#     subfolder=$(echo "$result" | awk -F'/' '{print $4}')
#     name="eval_results/rebuttal/pickscore/${subfolder}"
#     echo "Evaluating $name"
#     CUDA_VISIBLE_DEVICES=0 python evaluator.py \
#         benchmark_type="pickscore" \
#         image_dir=$result \
#         prompt_dir="datasets/eval_prompts/partiprompts.json" \
#         name=$name
#     wait
# done

#!/usr/bin/env bash
set -u

results=(
  # "main_results/rebuttal/ablate_ratio/50_50/random_seed_999"
  # "main_results/rebuttal/ablate_ratio/75_25/random_seed_999"
  # "main_results/rebuttal/increase_labeled_size/100_4900/random_seed_999"
  # "main_results/rebuttal/increase_labeled_size/200_4800/random_seed_999"
  # "main_results/rebuttal/increase_labeled_size/500_4500/random_seed_999"
  # "main_results/rebuttal/small_synthetic_portion/0.1_percent/random_seed_999"
  # "main_results/rebuttal/small_synthetic_portion/0.05_percent/random_seed_999"
  # "main_results/rebuttal/dpo_50_50/checkpoint-100/random_seed_999"
  # "main_results/rebuttal/dpo_75_25/random_seed_999"
  # "main_results/rebuttal/dpo_100/checkpoint-100/random_seed_999"
  # "main_results/rebuttal/dpo_200/checkpoint-100/random_seed_999"
  # "main_results/rebuttal/dpo_500/checkpoint-100/random_seed_999"
  # "main_results/rebuttal/dpo_1250/checkpoint-100/random_seed_999"
  # "main_results/rebuttal/dpo_75_25/checkpoint-100/random_seed_999"
  # "main_results/rebuttal_dpo/10_percent/dpo/random_seed_999"  
  # "main_results/rebuttal_dpo/20_percent/dpo/random_seed_999"
  # "main_results/rebuttal_dpo/30_percent/dpo/random_seed_999"
  # "main_results/rebuttal_dpo/40_percent/dpo/random_seed_999"
  # "main_results/rebuttal_dpo/50_percent/dpo/random_seed_999"
  # "main_results/rebuttal_dpo/dpo_5k_25_75_animated/random_seed_999"
  # "main_results/rebuttal/dr_5k_25_75_animated/random_seed_999"
  "main_results/rebuttal_dpo/dr_5k_25_75_animated/random_seed_999"
)

# GPUs you want to use for evaluation (edit as needed)
GPUS=(0 1 2 3 4 5)
# GPUS=(3 4 5 6 7)
MAX_JOBS=${#GPUS[@]}

# mkdir -p "eval_results/rebuttal/pickscore"

job_i=0
for result in "${results[@]}"; do
  # Extract 4th term: main_results/rebuttal/<term3>/<term4>/random_seed_999 -> term4
  subfolder=$(echo "$result" | awk -F'/' '{print $3}')
  name="eval_results/rebuttal_dpo/pickscore/${subfolder}"
  gpu="${GPUS[$((job_i % MAX_JOBS))]}"

  echo "Evaluating ${result} -> ${name} on GPU ${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" python evaluator.py \
    benchmark_type="pickscore" \
    image_dir="$result" \
    prompt_dir="datasets/eval_prompts/partiprompts.json" \
    name="$name" &

  job_i=$((job_i + 1))

  # Keep at most one job per GPU running at a time
  if (( job_i % MAX_JOBS == 0 )); then
    wait
  fi
done

# Wait for the last batch
wait
echo "All evaluations finished."
