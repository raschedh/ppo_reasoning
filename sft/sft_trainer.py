import os
import argparse
import torch
import torch.distributed as dist
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import random
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", required=True)
    p.add_argument("--model_path", default="Qwen/Qwen2.5-7B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--max_seq_length", type=int, default=4096)  # Your max token limit
    return p.parse_args()

def formatting_prompts_func(examples):
    return [
        f"{examples['query'][i]}\n\n### RESPONSE:\n{examples['completion'][i]}"
        for i in range(len(examples["query"]))
    ]

class PrintExampleCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, log_file):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.log_file = log_file

    def on_log(self, args, state, control, **kwargs):
        # Trigger every logging_steps
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            # Pick a random example
            example = random.choice(self.dataset)
            query = example["query"]

            # Tokenize and generate output
            inputs = self.tokenizer(query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            model = kwargs["model"]
            model.eval()
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_seq_length - inputs["input_ids"].shape[1], do_sample=False)
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            log_entry = (
                "\n" + "=" * 50 +
                f"\nStep {state.global_step}\n"
                f"QUERY:\n{query}\n\nMODEL OUTPUT:\n{generated_text}\n" +
                "=" * 50 + "\n"
            )

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)

def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir.rstrip('/')}_{timestamp}"
    log_file = f"training_examples_{timestamp}.log"

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    train_dataset = load_dataset(args.train_path)["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_dtype)

    # Response template for masking completions only
    response_template = "### RESPONSE:\n"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=500,
        logging_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=True,
        report_to=None,
        optim="adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        packing=False,  # ✅ Must be False for completion-only masking
    )

    # LoRA configuration
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,  # ✅ Required for completion-only
        callbacks=[PrintExampleCallback(tokenizer=tokenizer, dataset=train_dataset, log_file=log_file)],  # ✅ Added callback

    )

    # Distributed setup if needed
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    # ✅ Evaluate initial performance before training
    model.eval()

    first_query = train_dataset[0]["query"]
    inputs = tokenizer(first_query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_seq_length - inputs["input_ids"].shape[1], do_sample=False)
    first_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # --- Random example ---
    random_query = random.choice(train_dataset)["query"]

    inputs = tokenizer(random_query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_seq_length - inputs["input_ids"].shape[1], do_sample=False)
    random_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # --- Log both to file ---
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            "\n" + "=" * 50 +
            "\n=== INITIAL MODEL CHECK (FIRST EXAMPLE) ===\n"
            f"QUERY:\n{first_query}\n\nMODEL OUTPUT:\n{first_output}\n" +
            "=" * 50 +
            "\n=== INITIAL MODEL CHECK (RANDOM EXAMPLE) ===\n"
            f"QUERY:\n{random_query}\n\nMODEL OUTPUT:\n{random_output}\n" +
            "=" * 50 + "\n"
        )

    # Train the model
    trainer.train()

    # Save the merged model
    peft_model = trainer.model
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
