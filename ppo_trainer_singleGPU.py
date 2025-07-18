import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import random
import time
import requests
from vllm import LLM, SamplingParams
from datasets import load_from_disk
# from simple_download import download_tldr_hf

# ---------- Global Log Files ---------- #
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"ppo_training_example_log_{RUN_TIMESTAMP}.txt"
# download_tldr_hf()

# ---------- Argument Parsing ---------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_path", required=True)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="./ppo_results")
    p.add_argument("--learning_rate", type=float, default=1.41e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--mini_batch_size", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--max_prompt_length", type=int, default=4096)

    return p.parse_args()


# ---------- Reward Function (ThinkPRM-based) ---------- #
def get_rewards(tokenizer, llm, problems, prefixes):
    """
    Batched reward computation using vLLM.
    Returns a list of floats (one per input).
    - If the policy tries to cheat by including ThinkPRM-style tokens, it gets -1.0 immediately.
    - Otherwise, reward is based purely on ThinkPRM step evaluations (boxed{correct}/boxed{incorrect}).
    """
    prompts = []
    for problem, prefix in zip(problems, prefixes):
        prompt = f"""You are given a math problem and a proposed step-by-step solution:
        [Math Problem]
        {problem}
        [Solution]
        {prefix}
        Review and critique each step in the proposed solution to determine whether each step is correct. 
        If the solution is incomplete, only verify the provided steps.
        """
        prompt = tokenizer.apply_chat_template(
            [{'role': "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        ) + "\nLet's verify step by step:"
        prompts.append(prompt)

    # ✅ Batched vLLM call
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    outputs = llm.generate(prompts, sampling_params)

    rewards = []
    for prefix, out in zip(prefixes, outputs):
        # ✅ Immediate penalty for reward hacking
        if "boxed{correct}" in prefix or "boxed{incorrect}" in prefix:
            rewards.append(-1.0)
            continue

        # ✅ Standard ThinkPRM-based reward
        text = out.outputs[0].text.strip()
        correct_steps = text.count("boxed{correct}")
        incorrect_steps = text.count("boxed{incorrect}")
        total_steps = correct_steps + incorrect_steps

        if total_steps == 0:
            reward = 0.0  # No explicit evaluation
        else:
            reward = (correct_steps - incorrect_steps) / total_steps
            reward = max(-1.0, min(1.0, reward))  # Clamp to [-1,1]

        rewards.append(reward)

    return rewards, outputs


# ---------- Logging Functions ---------- #
def log_step(step, queries, responses, rewards, reward_outputs):
    avg_reward = sum(rewards) / len(rewards)
    
    # ---- Pick a random example ----
    idx = random.randint(0, len(queries) - 1)
    critique = reward_outputs[idx].outputs[0].text.strip()

    # ---- Write to single log file ----
    log_entry = (
        f"\n[Step {step} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
        f"Average Reward: {avg_reward:.4f}\n\n"
        f"Prompt:\n{queries[idx]}\n\n"
        f"Response:\n{responses[idx]}\n\n"
        f"Reward: {rewards[idx]:.4f}\n\n"
        f"Reward Model Output:\n{critique}\n"
        + "=" * 50 + "\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
            f"Step {step}, Avg Reward: {avg_reward:.4f}\n"
        )
        f.write(log_entry)

# ---------- Main PPO Training ---------- #
def main():
    args = parse_args()
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ✅ Load Dataset
    # dataset = load_dataset(args.dataset)["train"]
    dataset = load_from_disk('data_hf/countdown_dataset')["train"]

    dataset = dataset.map(lambda x: {
        "prompt": (
            f"""Using the numbers {str(x['nums'])}, create an equation that equals {str(x['target'])}.
            You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags"""
        )
    })
    dataset = dataset.remove_columns(["target", "nums"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ✅ Tokenizer
    gen_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id
    gen_tokenizer.padding_side = "left"

    # ✅ Load SFT Model with LoRA
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # first way 
    sft_model = get_peft_model(sft_model, lora_config)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)

    # second way
    # sft_model.gradient_checkpointing_enable()
    # policy = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model, trust_remote_code=True, peft_config=lora_config)

    # ✅ PPO Config
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        log_with=None,
        init_kl_coef=0.2,
        target_kl=6.0,
    )
    trainer = PPOTrainer(config=ppo_config, model=policy, tokenizer=gen_tokenizer)

    # ✅ Reward Model (ThinkPRM)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    reward_llm = LLM(model=args.reward_model_path, max_model_len=16384, gpu_memory_utilization=0.3)

    # ✅ PPO Training Loop
    for step, batch in enumerate(tqdm(dataloader, desc="PPO Training", unit="batch")):
        if step >100:
            break
        queries = batch["prompt"]

        tokenized_queries = gen_tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_prompt_length
        ).to(trainer.accelerator.device)

        queries_tensors = [q for q in tokenized_queries.input_ids]

        # ✅ Generate responses
        response_ids = trainer.generate(
            queries_tensors,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=gen_tokenizer.eos_token_id
        )

        response_texts = gen_tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        for i, txt in enumerate(response_texts):
            print(f"[Sample {i}] {repr(txt)}")
        # ✅ Compute rewards
        rewards, reward_outputs = get_rewards(reward_tokenizer, reward_llm, queries, response_texts)
        rewards = [torch.tensor(r, dtype=torch.float32).to(trainer.accelerator.device) for r in rewards]
        
        # ✅ Convert to tensors before PPO step
        tokenized_responses = gen_tokenizer(
            response_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_prompt_length
        ).to(trainer.accelerator.device)

        responses_tensors = [r for r in tokenized_responses.input_ids]

        trainer.step(queries_tensors, responses_tensors, rewards)

        # ✅ Logging
        if step % 5 == 0:
            log_step(step, queries, response_texts, rewards, reward_outputs)

    # ✅ Save Model
    policy.save_pretrained(args.output_dir)
    gen_tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] PPO Training Completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
