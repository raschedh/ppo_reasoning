#!/bin/bash
#SBATCH --job-name=ppo_qwen
#SBATCH --output=logs/%A_ppo_qwen.log
#SBATCH --gres=gpu:2

source ~/.bashrc
# conda activate your_env
pip install --upgrade --quiet transformers==4.38.2 trl==0.9.4 peft==0.10.0 datasets
pip install --upgrade vllm

# -------- Paths & Config -------- #
SFT_MODEL="<insert SFT Model>"
REWARD_MODEL="launch/ThinkPRM-7B"
DATASET="Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR="ppo_model/"
PORT=8000
TASK="reward"
REWARD_MODEL_URL="http://localhost:${PORT}/v1/chat/completions"

echo "[INFO] Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# -------- Start vLLM server on GPU 0 -------- #
echo "[INFO] Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python vllm_prm.py \
    --model ${REWARD_MODEL} \
    --port ${PORT} \
    --gpu-memory-utilization 0.6 \
    --task "reward" \
    --max_model_len 4096 &

VLLM_PID=$!

# -------- Run PPO training on GPU 1 -------- #
echo "[INFO] Starting PPO training on GPU 1..."

CUDA_VISIBLE_DEVICES=1 python -u ppo_trainer_multi.py \
    --sft_model_path ${SFT_MODEL} \
    --reward_model_path ${REWARD_MODEL} \
    --reward_model_url ${REWARD_MODEL_URL} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --mini_batch_size 2\
    --batch_size 2 \
    --ppo_epochs 1

# -------- Cleanup -------- #
echo "[INFO] Stopping vLLM server..."
kill ${VLLM_PID}

echo "[INFO] PPO training finished."
