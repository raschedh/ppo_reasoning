#!/bin/bash
#SBATCH --job-name=vllm_thinkprm
#SBATCH --output=logs/%A_vllm_thinkprm.log
#SBATCH --gres=gpu:1

source ~/.bashrc
# conda activate your_env

MODEL="launch/ThinkPRM-1.5B"
PORT=8000

echo "[INFO] Starting vLLM serve for Process Reward Model..."
python vllm_prm.py \
    --model ${MODEL} \
    --port ${PORT} \
    --max_model_len 16384 \
    --gpu_util 0.6

echo "[INFO] Server stopped."
