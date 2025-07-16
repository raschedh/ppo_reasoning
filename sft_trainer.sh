#!/bin/bash
#SBATCH --job-name=sft_trainer
#SBATCH --output=logs/%A_sft_trainer.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
source ~/.bashrc

# -------- User-editable ---------------------------------------------- #
MODEL_PATH="Qwen/Qwen2.5-7B"  # Base model path
TRAIN_PATH="Asap7772/cog_behav_all_strategies"  # Training data
OUTPUT_DIR="sft_model/"
mkdir -p "${OUTPUT_DIR}"

PER_DEVICE_BATCH_SIZE=1
GRAD_ACC=5
EPOCHS=2
LR=3e-4
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LOG_STEPS=1
# --------------------------------------------------------------------- #

echo "Launching SFT run on $(hostname)..."

accelerate launch \
    --num_processes 2 \
    train_SFT.py \
    --train_path "${TRAIN_PATH}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    --logging_steps ${LOG_STEPS}

echo "Job finished."
