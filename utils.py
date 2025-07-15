import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(save_dir="Qwen2.5-7B"):
    os.makedirs(save_dir, exist_ok=True)

    # Download tokenizer and model directly from HF hub
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

    # Save locally
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"✅ Model and tokenizer saved to: {save_dir}")
    return 

def load_model(save_dir="Qwen2.5-7B"):
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForCausalLM.from_pretrained(save_dir)
    print("✅ Model loaded successfully from local directory!")
    return model, tokenizer

if __name__ == "__main__":
    # get_model()
    model, _ = load_model()
    print(model)
