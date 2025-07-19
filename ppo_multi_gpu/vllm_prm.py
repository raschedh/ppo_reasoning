import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--task", type=str, default="reward")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    print(f"🚀 Starting vLLM server for {args.model} on port {args.port}")

    subprocess.run([
        "vllm", "serve",
        args.model,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--task", args.task,
        "--trust-remote-code"
    ])

if __name__ == "__main__":
    main()