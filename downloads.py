import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def download_model(model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to: {save_dir}")
    return 

def download_dataset(dataset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset(dataset_name)
    ds.save_to_disk(save_dir)
    print(f"Dataset saved locally at: {save_dir}")
    return


if __name__ == "__main__":
    download_model(model_name="launch/ThinkPRM-1.5B", save_dir="ThinkPRM-1.5B")
    download_model(model_name="Qwen/Qwen2.5-7B", save_dir="Qwen2.5-7B")
    
    download_dataset("Asap7772/cog_behav_all_strategies", "cog_behav_all_strategies")
    download_dataset("Jiayi-Pan/Countdown-Tasks-3to4", "Countdown-Tasks-3to4")