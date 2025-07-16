import argparse
from vllm import serve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="launch/ThinkPRM-1.5B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_util", type=float, default=0.6)
    args = parser.parse_args()

    print(f"🚀 Starting vLLM server for {args.model} on port {args.port}")
    serve.main(
        model=args.model,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True
    )

if __name__ == "__main__":
    main()
