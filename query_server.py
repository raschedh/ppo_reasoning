import requests

url = "http://localhost:8000/v1/completions"
payload = {
    "model": "launch/ThinkPRM-7B",
    "prompt": [
        "You are given a math problem and a proposed step-by-step solution:\n"
        "[Math Problem]\n"
        "Using the numbers 3, 5, 8, create an equation that equals 24.\n"
        "[Solution]\n"
        "Step 1: 8 - 5 = 3.\nStep 2: 3 * 3 = 9.\nStep 3: 9 + 5 = 14.\n"
        "Let's verify step by step:"
    ],
    "max_tokens": 512,
    "temperature": 0.0
}

r = requests.post(url, json=payload)
print(r.json())
