#!/bin/bash
#SBATCH --job-name=ppo_qwen
#SBATCH --output=logs/%A_ppo_qwen.log
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

source ~/.bashrc
# conda activate your_rlhf_env

SFT_MODEL="models/qwen2.5-7b_sft"
DATASET="data/new_math_prompts.json"
OUTPUT_DIR="models/ppo_qwen2.5"

echo "[INFO] Starting PPO training..."
python ppo_train.py \
    --sft_model_path ${SFT_MODEL} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --ppo_epochs 4

echo "[INFO] PPO training finished."
