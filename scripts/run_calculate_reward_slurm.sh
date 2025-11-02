#!/bin/bash
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --job-name=calc_reward
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/calc_reward_%A_%a.out
#SBATCH --error=logs/calc_reward_%A_%a.err
#SBATCH --array=0-99  # Run 100 jobs in parallel (adjust as needed)
#SBATCH --requeue  # Auto-requeue on preemption

# Print compute node info
echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "================="

# Set default values
DATASET_NAME="sayakpaul/pickapic_v2_webdataset"
OUTPUT_DIR="datasets/pickscore_results"
SEED=42
NUM_JOBS=100  # Should match the number in --array

# Create logs directory
mkdir -p logs

# Run the script for this array job with resume capability
python calculate_reward.py \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_jobs $NUM_JOBS \
    --seed $SEED \
    --resume
