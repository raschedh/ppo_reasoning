import os
import argparse
import torch
import torch.distributed as dist
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from utils import get_model
from datasets import load_from_disk

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--model_path", default="Qwen2.5-7B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--max_seq_length", type=int, default=4096)  # Allow full Qwen2.5 context
    return p.parse_args()

def formatting_prompts_func(examples):
    """
    Format prompt-completion pairs for completion-only fine-tuning.
    Use a unique marker before the completion for proper masking.
    """
    output_texts = []
    for i in range(len(examples["prompt"])):
        text = f"{examples['prompt'][i]}\n\n### RESPONSE:\n{examples['completion'][i]}"
        output_texts.append(text)
    return output_texts

def main() -> None:
    args = parse_args()

    train_dataset = load_from_disk(args.train_file)
    model, tokenizer = get_model(save_dir=args.model_path)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_seq_length=min(args.max_seq_length, tokenizer.model_max_length),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=500,
        save_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=None,
        optim="adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        packing=True,  
        response_template="### RESPONSE:\n",  
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
    )

    # Distributed setup if needed
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    # Train the model
    trainer.train()

    # Save the merged model
    peft_model = trainer.model
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
