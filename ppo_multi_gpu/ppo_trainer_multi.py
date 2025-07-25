import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import time
import requests
from datetime import datetime

# ---------- Global Log Files ---------- #
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"ppo_training_example_log_{RUN_TIMESTAMP}.txt"

# -------------------- PPO Training -------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_path", required=True)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--reward_model_url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", default="./ppo_results")
    p.add_argument("--learning_rate", type=float, default=1.41e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--mini_batch_size", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    return p.parse_args()

# -------------------- Server Wait -------------------- #
def wait_for_vllm_server(url, timeout=300):
    print(f"[INFO] Waiting for vLLM server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url.replace("/score", "/health"))
            if r.status_code == 200:
                print("[INFO] 🚀 vLLM server is ready!")
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server not responding after {timeout} seconds.")

# -------------------- ThinkPRM Reward -------------------- #
def get_rewards(tokenizer, prompts, responses, reward_model_path, reward_model_url):
    
    rewards = []
    outputs = []
    
    for problem, prefix in zip(prompts, responses):

        # if the model tries to reward hack, we automatically assign the worst reward
        if any(x in prefix for x in ["boxed{correct}", "boxed{incorrect}",
                                     "Is the solution correct? Yes", "Is the solution correct? No"]):
            rewards.append(-1.0)
            outputs.append(prefix)
            continue
    

        prompt = f"""You are given a math problem and a proposed step-by-step solution:
            [Math Problem]
            {problem}
            [Solution]
            {prefix}
            Review and critique each step in the proposed solution to determine whether each step is correct. 
            If the solution is incomplete, only verify the provided steps.
            """
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        ) + "\nLet's verify step by step:"

        payload = {
            "model": reward_model_path,  
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }

        r = requests.post(reward_model_url, json=payload)
        r.raise_for_status()
        print(r.json())
        verification_cot = r.json()["choices"][0]["message"]["content"]

        correct_steps = verification_cot.count("\\boxed{correct}")
        incorrect_steps =  verification_cot.count("\\boxed{incorrect}")
        total_steps = correct_steps + incorrect_steps
        
        outputs.append(verification_cot)

        if 20 >= total_steps > 0:
            base_reward = (correct_steps - incorrect_steps) / total_steps

            if "Is the solution correct? Yes" in verification_cot:
                base_reward += 0.2
            elif "Is the solution correct? No" in verification_cot:
                base_reward -= 0.2
            
            reward = max(-1.0, min(1.0, base_reward))
            rewards.append(reward)

        else:
            # we don't want really long chains for problems like this
            rewards.append(-1.0)

    return rewards, outputs

# ---------- Logging Functions ---------- #
def log_step(step, query, response, reward, reward_output, avg_reward):
    critique = reward_output.outputs[0].text.strip()

    log_entry = (
        f"\n[Step {step}\n"
        f"Average Reward: {avg_reward:.4f}\n\n"
        f"Prompt:\n{query}\n\n"
        f"Response:\n{response}\n\n"
        f"Reward: {reward:.4f}\n\n"
        f"Reward Model Output:\n{critique}\n"
        + "=" * 50 + "\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

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

    gen_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id
    gen_tokenizer.padding_side = "left"

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)

    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    sft_model = get_peft_model(sft_model, lora_config)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)

    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        log_with=None, init_kl_coef=0.2, target_kl=6.0,
    )
    trainer = PPOTrainer(config=ppo_config, model=policy, tokenizer=gen_tokenizer)

    for step, batch in enumerate(tqdm(dataloader, desc="PPO Training", unit="batch")):
        # remove for entire dataset
        if step > 50:
            break
        queries = batch["prompt"]

        tokenized_queries = gen_tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_prompt_length
        ).to(trainer.accelerator.device)

        queries_tensors = [q for q in tokenized_queries.input_ids]

        response_ids = trainer.generate(
            queries_tensors,
            max_length=args.max_prompt_length,
            do_sample=True,
            temperature=0.3,
            pad_token_id=gen_tokenizer.eos_token_id
        )

        response_texts = gen_tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Compute rewards
        rewards, reward_outputs = get_rewards(reward_tokenizer, queries, response_texts, args.reward_model_path, args.reward_model_url)
        rewards = [torch.tensor(r, dtype=torch.float32).to(trainer.accelerator.device) for r in rewards]
        
        tokenized_responses = gen_tokenizer(
            response_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_prompt_length
        ).to(trainer.accelerator.device)

        responses_tensors = [r for r in tokenized_responses.input_ids]

        trainer.step(queries_tensors, responses_tensors, rewards)

        # Logging
        if step % 5 == 0:
            log_step(step, queries[0], response_texts[0], rewards[0], reward_outputs[0], sum(rewards) / len(rewards))
        
        del tokenized_queries, tokenized_responses, queries_tensors, responses_tensors, response_ids
        del rewards, reward_outputs, response_texts
        torch.cuda.empty_cache()

    policy.save_pretrained(args.output_dir)
    gen_tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] PPO Training Completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
