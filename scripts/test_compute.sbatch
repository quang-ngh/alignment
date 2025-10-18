#!/bin/bash

# zabih has 6 gpus, 32 cpus and 512GB ram.

###cat /share/zabih/g2_usage/sacct
###SBATCH --constraint="[r6000|a5000|a6000|3090|a100|a40]"
###SBATCH --constraint=zabih|gpu-high|gpu-mid
###SBATCH --constraint=zabih|gpu-high
###SBATCH --partition=zabih

#SBATCH --partition=zabih
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

# Start an interactive session
sleep 1d