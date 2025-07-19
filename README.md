# SFT and PPO for Basic Reasoning Model

This repo contains implementations for training a basic reasoning-style model using **Supervised Fine-Tuning (SFT)** (LoRA) and **Proximal Policy Optimization (PPO)**. 

There are two main files:
1. **Single GPU** SFT/PPO training - the Process Reward Model (PRM - [ThinkPRM](https://github.com/mukhal/thinkprm) 1.5B parameters) and policy ([Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B)) are on the same GPU.
2. **Multi-GPU** SFT/PPO training - the PRM (larger ThinkPRM 7B) is deployed as a vLLM server and the policy (Qwen2.5 7B) is on another GPU making inference calls. This particular PRM doesn't support certain endpoints so it is somewhat experimental. 

The above were tested using Nvidia A40 GPUs.


---
## Project Overview

### `sft`
- SFT for the initial reasoning policy. 
- Fine-tunes it on reasoning dataset (Countdown Maths task - make target number from 3 others) to initialise PPO training. We use a [huggingface dataset](https://huggingface.co/datasets/Asap7772/cog_behav_all_strategies).

### ** `ppo_single_gpu`**
- PPO training loop where both the **policy model** and **PRM** run on a **single GPU**.
- Suitable for smaller-scale experiments

### ** `ppo_multi_gpu`**
- PPO training with 2 GPUs.
- The **PRM** is accessed remotely through a HTTP call to a vLLM server.
- Allows scaling to larger PRMs while keeping the policy model training isolated.

## Libraries

```bash
torch
transformers
datasets
tqdm
trl
peft
requests
vllm
```
