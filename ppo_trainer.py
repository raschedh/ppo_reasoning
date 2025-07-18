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

from openai import OpenAI
from model_utils.io_utils import prepare_input, derive_step_rewards_vllm

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
LOG_EXAMPLE_FILE = f"ppo_training_example_log_{timestamp}.txt"
LOG_REWARD_FILE = f"ppo_training_reward_log_{timestamp}.txt"

# -------------------- PPO Training -------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_path", required=True)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--reward_model_url", default="http://localhost:8081/v1")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="./ppo_results")
    p.add_argument("--learning_rate", type=float, default=1.41e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--mini_batch_size", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    # p.add_argument("--max_gen_length", type=int, default=4096)
    return p.parse_args()

# -------------------- vLLM PRM -------------------- #
def wait_for_vllm_server(url, timeout=300):
    print(f"[INFO] Waiting for vLLM server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url.replace("/v1", "/health"))
            if r.status_code == 200:
                print("[INFO] 🚀 vLLM server is ready!")
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server not responding after {timeout} seconds.")

def get_rewards_skywork(tokenizer, prompts, responses, reward_model_url):
    processed_data = [
        prepare_input(prompt, response, tokenizer=tokenizer, step_token="\n")
        for prompt, response in zip(prompts, responses)
    ]
    input_ids, steps, reward_flags = zip(*processed_data)

    client = OpenAI(api_key="EMPTY", base_url=reward_model_url)
    model_name = client.models.list().data[0].id
    rewards = client.embeddings.create(input=input_ids, model=model_name)
    step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
    scalar_rewards = [sum(sr) / len(sr) for sr in step_rewards]
    return scalar_rewards

# def get_rewards_thinkprm(prompts, responses, reward_model_path, reward_model_url):
#     tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

#     rewards = []
#     for prompt, response in zip(prompts, responses):
#         formatted_prompt = f"""You are given a math problem and a proposed step-by-step solution:
#         [Math Problem]
#         {prompt}
#         [Solution]
#         {response}
#         Review and critique each step in the proposed solution to determine whether each step is correct. 
#         If the solution is incomplete, only verify the provided steps."""

#         formatted_prompt = tokenizer.apply_chat_template(
#             [{'role': "user", "content": formatted_prompt}],
#             tokenize=False,
#             add_generation_prompt=True
#         ) + "\nLet's verify step by step:"

#         payload = {
#             "model": reward_model_path,
#             "prompt": [formatted_prompt],
#             "max_tokens": 4096,
#             "temperature": 0.0
#         }

#         r = requests.post(reward_model_url, json=payload)
#         r.raise_for_status()
#         print(formatted_prompt)
#         verification_cot = r.json()["choices"][0]["text"]

#         correct_steps = verification_cot.count("\\boxed{correct}")
#         total_steps = correct_steps + verification_cot.count("\\boxed{incorrect}")
#         rewards.append(correct_steps / total_steps if total_steps > 0 else 0.0)

#     return rewards

# -------------------- Logging -------------------- #
def log_example(step, queries, responses, rewards):
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
    with open(LOG_EXAMPLE_FILE, "a") as f:
        f.write(log_entry)

def log_all_rewards(step, rewards):
    avg_reward = sum(rewards) / len(rewards)
    with open(LOG_REWARD_FILE, "a") as f:
        f.write(f"{timestamp}, Step {step}, Avg Reward: {avg_reward:.4f}\n")

# -------------------- Main -------------------- #
def main():
    args = parse_args()

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    wait_for_vllm_server(args.reward_model_url)

    dataset = load_dataset(args.dataset)["train"]
    dataset = dataset.map(lambda x: {
        "prompt": (
            f"Using the numbers {str(x['nums'])}, create an equation that equals {str(x['target'])}..."
        )
    })
    dataset = dataset.remove_columns(["target", "nums"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ✅ Generation tokenizer (for SFT model & PPO)
    gen_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id
    gen_tokenizer.padding_side = "left"

    # ✅ PRM tokenizer (for reward computation only)
    prm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    sft_model = get_peft_model(sft_model, lora_config)
    policy = AutoModelForCausalLMWithValueHead(sft_model)

    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        log_with=None, init_kl_coef=0.2, target_kl=6.0,
    )
    trainer = PPOTrainer(config=ppo_config, model=policy, tokenizer=gen_tokenizer)

    # ✅ PPO Training Loop
    for step, batch in enumerate(tqdm(dataloader, desc="PPO Training", unit="batch")):
        queries = batch["prompt"]

        tokenized_queries = gen_tokenizer(
            queries, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_prompt_length
        )
        queries_tensors = [q for q in tokenized_queries.input_ids]
        response_ids = trainer.generate(
            queries_tensors, max_length=args.max_prompt_length, do_sample=True, temperature=0.7
        )
        response_texts = gen_tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # ✅ Reward computation via vLLM PRM
        rewards = get_rewards_skywork(prm_tokenizer, queries, response_texts, args.reward_model_url)

        trainer.step(queries, response_texts, rewards)
        avg_reward = sum(rewards) / len(rewards)
        print(f"[DEBUG] Step {step} completed. Avg Reward: {avg_reward:.4f}")

        log_all_rewards(step, rewards)
        if step % 50 == 0:
            log_example(step, queries, response_texts, rewards)

    policy.save_pretrained(args.output_dir)
    gen_tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
