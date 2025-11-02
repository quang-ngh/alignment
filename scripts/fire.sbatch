#!/bin/bash

# zabih has 6 gpus, 32 cpus and 512GB ram.

###cat /share/zabih/g2_usage/sacct
###SBATCH --constraint="[r6000|a5000|a6000|3090|a100|a40]"
###SBATCH --constraint=zabih|gpu-high|gpu-mid
###SBATCH --constraint=zabih|gpu-high
###SBATCH --partition=zabih

#SBATCH --partition=zabih
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

# Start an interactive session
sleep 1d

# source ~/.bashrc
# source activate simpro
# cd SimPro/BOAT
# python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 150 --imb-ratio-unlabel 150 --tau 1.0 --ema-u 0.99 --tau3 2.0 --tau4 1.0
# 
