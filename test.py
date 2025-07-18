import requests
from transformers import AutoTokenizer

model_id = "launch/ThinkPRM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

problem = "Solve for x: 2x + 3 = 7"
prefix = "Step 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide by 2: x = 1"

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

payload = {
    "model": model_id,
    "prompt": prompt,
    "max_tokens": 512,
    "temperature": 0.0
}

r = requests.post("http://localhost:8000/v1/completions", json=payload)
print(r.json()["choices"][0]["text"])
