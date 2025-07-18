#!/bin/bash
#SBATCH --job-name=ppo
#SBATCH --output=logs/%A_ppo.log

source ~/.bashrc
# pip install --upgrade --quiet transformers==4.38.2 trl==0.9.4 peft==0.10.0 datasets accelerate
# pip install --upgrade vllm
# huggingface-cli login

# -------- Paths & Config -------- #
SFT_MODEL="seldschuk/sft-model"
REWARD_MODEL="launch/ThinkPRM-1.5B"
DATASET="Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR="ppo_model/"

python -u ppo_trainer_singleGPU.py \
    --sft_model_path ${SFT_MODEL} \
    --reward_model_path ${REWARD_MODEL} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --mini_batch_size 2\
    --batch_size 2 \
    --ppo_epochs 1

echo "[INFO] PPO training finished."
