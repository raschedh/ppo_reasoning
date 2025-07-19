# PPO and SFT Training for Reasoning Models

This repository contains implementations for training reasoning models using **Supervised Fine-Tuning (SFT)** and **Proximal Policy Optimization (PPO)**. It supports two configurations:

1. **Single-GPU Training** – PRM (Pre-trained Reward Model) and policy model run on the same GPU.
2. **Multi-GPU / Distributed Training** – PRM served via a vLLM server while the policy model runs on a separate GPU and communicates with the PRM.

---

## 📂 Code Overview

### **1. `sft_trainer.py`**
- Implements **Supervised Fine-Tuning** for the initial reasoning policy.
- Loads a base model (e.g., LLaMA, GPT-style transformer).
- Fine-tunes it on reasoning or chain-of-thought datasets to provide a strong initialization for PPO training.

### **2. `ppo_trainer_singleGPU.py`**
- PPO training loop where both the **policy model** and **PRM** run on a **single GPU**.
- Uses standard PPO components: rollout collection, advantage estimation, and clipped policy updates.
- Suitable for smaller-scale experiments or hardware-limited setups.

### **3. `ppo_trainer.py`**
- PPO training with **two GPUs** or distributed setup.
- The **PRM** is accessed remotely through an HTTP/gRPC call to a vLLM server (see below).
- Allows scaling to larger PRMs while keeping the policy model training isolated.

### **4. `vllm_prm.py`**
- Starts a **vLLM-based server** hosting the Pre-trained Reward Model (PRM).
- Provides an API endpoint for PPO training to query rewards.
- Ideal for decoupling PRM evaluation from the policy optimization.

---

## 🏗️ Overall Process

1. **Supervised Fine-Tuning (SFT)**
   - Run `sft_trainer.py` to fine-tune the base reasoning model on high-quality reasoning datasets.
   - The output is the initial policy used for PPO.

2. **PPO Training**
   - **Option 1: Single GPU**
     - Run `ppo_trainer_singleGPU.py`.
     - Both policy and PRM share a single GPU.
   - **Option 2: Multi-GPU / vLLM**
     - Start the PRM server:
       ```bash
       python vllm_prm.py
       ```
     - Run PPO training with remote PRM evaluation:
       ```bash
       python ppo_trainer.py
       ```
     - The policy model communicates with the PRM server to compute rewards.

3. **Inference / Evaluation**
   - After training, the final policy model can be used for reasoning tasks or further evaluation.

---

## 🔧 Models Used

- **Policy Model:** A transformer-based reasoning model (e.g., LLaMA, GPT-J, or similar).
- **PRM (Reward Model):** A pre-trained model fine-tuned to score reasoning quality.
- **SFT Dataset:** Chain-of-thought or reasoning-focused dataset (e.g., CoT, PRM800K, etc.).

The specific models and datasets can be swapped by updating the model checkpoints and data paths in the trainers.

---

## 🚀 Getting Started

### **Install Dependencies**
```bash
pip install -r requirements.txt
