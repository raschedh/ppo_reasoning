import os
import time
import requests
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

# -------------------- Reward Model Query -------------------- #
def wait_for_vllm_server(url, timeout=300):
    """Wait until vLLM server is ready (default timeout: 5 min)."""
    print(f"[INFO] Waiting for vLLM server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url.replace("/v1/completions", "/health"))
            if r.status_code == 200:
                print("[INFO] vLLM server is ready!")
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server not responding after {timeout} seconds.")

def get_rewards(prompts, responses, reward_model_path, reward_model_url):
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

    rewards = []
    for prompt, response in zip(prompts, responses):
        formatted_prompt = f"""You are given a math problem and a proposed step-by-step solution:
        [Math Problem]
        {prompt}
        [Solution]
        {response}
        Review and critique each step in the proposed solution to determine whether each step is correct. 
        If the solution is incomplete, only verify the provided steps."""

        formatted_prompt = tokenizer.apply_chat_template(
            [{'role': "user", "content": formatted_prompt}],
            tokenize=False,
            add_generation_prompt=True
        ) + "\nLet's verify step by step:"

        payload = {
            "model": reward_model_path,
            "prompt": [formatted_prompt],
            "max_tokens": 4096,
            "temperature": 0.0
        }

        r = requests.post(reward_model_url, json=payload)
        r.raise_for_status()
        verification_cot = r.json()["choices"][0]["text"]

        correct_steps = verification_cot.count("\\boxed{correct}")
        total_steps = correct_steps + verification_cot.count("\\boxed{incorrect}")
        rewards.append(correct_steps / total_steps if total_steps > 0 else 0.0)

    return rewards

# -------------------- PPO Training -------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_path", required=True)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--reward_model_url", default="http://localhost:8000/v1/completions")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="./ppo_results")
    p.add_argument("--learning_rate", type=float, default=1.41e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--max_gen_length", type=int, default=4096)
    return p.parse_args()

def main():
    args = parse_args()

    # ✅ Ensure bf16 or fp16 on the current GPU
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}, dtype: {torch_dtype}")

    # ✅ Wait for vLLM server to be ready
    wait_for_vllm_server(args.reward_model_url)

    # ✅ Load dataset
    dataset = load_dataset("json", data_files=args.dataset)["train"]

    # ✅ Load tokenizer & policy model
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch_dtype
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ✅ Add value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    # ✅ LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)

    # ✅ PPO Config
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        log_with=None,
        init_kl_coef=0.2,
        target_kl=6.0,
    )

    trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    # ✅ PPO Training Loop
    for epoch, batch in enumerate(dataset):
        query = batch["prompt"]
        tokenized_query = tokenizer(query, return_tensors="pt", truncation=True, max_length=args.max_prompt_length).to(model.device)

        response_ids = trainer.generate(
            tokenized_query.input_ids,
            max_new_tokens=args.max_gen_length,
            do_sample=True,
            temperature=0.7
        )
        response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        rewards = get_rewards([query], response_texts, args.reward_model_path, args.reward_model_url)
        trainer.step([query], response_texts, rewards)
        print(f"[Epoch {epoch}] Reward: {rewards}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
