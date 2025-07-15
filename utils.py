import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_model(save_dir):
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForCausalLM.from_pretrained(save_dir)
    return model, tokenizer

def download_model(model_dir):
    os.makedirs(model_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"✅ Model and tokenizer saved to: {model_dir}")
    return 

def download_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    ds = load_dataset(data_dir)
    ds.save_to_disk(data_dir)
    print(f"✅ Dataset saved locally at: {data_dir}")
    return 

if __name__ == "__main__":
    download_model(model_dir="ThinkPRM-7B")
    download_model(model_dir="Qwen/Qwen2.5-7B")
    
    download_dataset(data_dir="Asap7772/cog_behav_all_strategies")
    download_dataset(data_dir="Jiayi-Pan/Countdown-Tasks-3to4")

