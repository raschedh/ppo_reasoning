import requests
from transformers import AutoTokenizer

MODEL = "launch/ThinkPRM-1.5B"
API_URL = "http://localhost:8000/v1/completions"  # replace with node if remote

# ThinkPRM example
problem = "Solve for x: 2x + 3 = 7"
prefix = "Step 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide by 2: x = 1"

# Format prompt EXACTLY as recommended
tokenizer = AutoTokenizer.from_pretrained(MODEL)
prompt = f"""You are given a math problem and a proposed step-by-step solution:
        [Math Problem]
        {problem}
        [Solution]
        {prefix}
        Review and critique each step in the proposed solution to determine whether each step is correct. 
        If the solution is incomplete, only verify the provided steps
        """

prompt = tokenizer.apply_chat_template(
    [{'role': "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) + "\nLet's verify step by step:"

# Send to vLLM server
payload = {
    "model": MODEL,
    "prompt": [prompt],
    "max_tokens": 4096,
    "temperature": 0.0
}

res = requests.post(API_URL, json=payload)
res.raise_for_status()
verification_cot = res.json()["choices"][0]["text"]

print("✅ Verification CoT:\n", verification_cot)
