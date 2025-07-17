import time
import requests
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import random
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
LOG_EXAMPLE_FILE = f"ppo_training_example_log_{timestamp}.txt"
LOG_REWARD_FILE = f"ppo_training_reward_log_{timestamp}.txt"

# -------------------- PPO Training -------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_path", required=True)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--reward_model_url", default="http://localhost:8000/v1/completions")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="./ppo_results")
    p.add_argument("--learning_rate", type=float, default=1.41e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--mini_batch_size", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--max_gen_length", type=int, default=4096)
    return p.parse_args()

# -------------------- Reward Model Query -------------------- #
def wait_for_vllm_server(url, timeout=300):
    """Wait until vLLM server is ready (default timeout: 5 min)."""
    print(f"[INFO] Waiting for vLLM server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url.replace("/v1/completions", "/health"))
            if r.status_code == 200:
                print("[INFO] 🚀 vLLM server is ready!")
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
        print(formatted_prompt)
        verification_cot = r.json()["choices"][0]["text"]

        correct_steps = verification_cot.count("\\boxed{correct}")
        total_steps = correct_steps + verification_cot.count("\\boxed{incorrect}")
        rewards.append(correct_steps / total_steps if total_steps > 0 else 0.0)

    return rewards

def log_example(step, queries, responses, rewards):
    """Log a random example with timestamp to a file."""
    idx = random.randint(0, len(queries) - 1)
    avg_reward = sum(rewards) / len(rewards)

    log_entry = (
        f"\nStep {step}\n"
        f"Average Reward: {avg_reward:.4f}\n"
        f"Prompt:\n{queries[idx]}\n\n"
        f"Response:\n{responses[idx]}\n\n"
        f"Reward: {rewards[idx]:.4f}\n"
        + "=" * 50 + "\n"
    )
    # Print to console and append to log file
    with open(LOG_EXAMPLE_FILE, "a") as f:
        f.write(log_entry)

def log_all_rewards(step, rewards):
    """Append average reward for every batch to a separate file."""
    avg_reward = sum(rewards) / len(rewards)
    with open(LOG_REWARD_FILE, "a") as f:
        f.write(f"{timestamp}, Step {step}, Avg Reward: {avg_reward:.4f}\n")

def main():
    args = parse_args()

    # ✅ Ensure bf16 or fp16 on the current GPU
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[INFO] Using device: {device}, dtype: {torch_dtype}")

    # ✅ Wait for vLLM server to be ready
    wait_for_vllm_server(args.reward_model_url)

    # ✅ Load dataset
    dataset = load_dataset(args.dataset)["train"]
    dataset = dataset.map(lambda x: {
        "prompt": (
            f"Using the numbers {str(x['nums'])}, create an equation that equals {str(x['target'])}... "
            )
        })
    dataset = dataset.remove_columns(["target", "nums"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ✅ Load tokenizer (base model tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ✅ Load your SFT model
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    # ✅ Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    base_model = get_peft_model(base_model, lora_config)

    # ✅ Wrap WITH the PPO value head (Correct Way)
    model = AutoModelForCausalLMWithValueHead(base_model)

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

    trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    # ✅ PPO Training Loop
    for step, batch in enumerate(dataloader):
        queries = batch["prompt"]  # already a list of strings

        # ✅ Tokenize batch (variable-length handled by padding=True)
        tokenized_queries = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length
        )

        queries_tensors = [q for q in tokenized_queries.input_ids]

        response_ids = trainer.generate(
            queries_tensors,
            max_new_tokens=args.max_gen_length,
            do_sample=True,
            temperature=0.7
        )
        response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # ✅ Get rewards (batch mode)
        rewards = get_rewards(queries, response_texts, args.reward_model_path, args.reward_model_url)

        # ✅ PPO step
        trainer.step(queries, response_texts, rewards)
        
        print(f"[DEBUG] Step {step} completed. Avg Reward: {sum(rewards)/len(rewards):.4f}")

        log_all_rewards(step, rewards)
        # ✅ Only log an example every 50 steps
        if step % 50 == 0:
            log_example(step, queries, response_texts, rewards)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
