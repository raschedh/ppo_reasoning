#!/bin/bash
#SBATCH --job-name=ppo_qwen
#SBATCH --output=logs/%A_ppo_qwen.log
#SBATCH --gres=gpu:2 
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/.bashrc
# conda activate your_rlhf_env

# -------- Paths & Config -------- #
SFT_MODEL="models/qwen2.5-7b_sft"
REWARD_MODEL="launch/ThinkPRM-7B"
DATASET="data/new_math_prompts.json"
OUTPUT_DIR="models/ppo_qwen2.5"
PORT=8000
REWARD_MODEL_URL="http://localhost:${PORT}/v1/completions"

echo "[INFO] Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# -------- Start vLLM server on GPU 0 -------- #
echo "[INFO] Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python vllm_prm.py \
    --model ${REWARD_MODEL} \
    --port ${PORT} \
    --max_model_len 16384 \
    --gpu_util 0.6 &

VLLM_PID=$!

# -------- Run PPO training on GPU 1 -------- #
echo "[INFO] Starting PPO training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python ppo_train.py \
    --sft_model_path ${SFT_MODEL} \
    --reward_model_path ${REWARD_MODEL} \
    --reward_model_url ${REWARD_MODEL_URL} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --ppo_epochs 4

# -------- Cleanup -------- #
echo "[INFO] Stopping vLLM server..."
kill ${VLLM_PID}

echo "[INFO] PPO training finished."
