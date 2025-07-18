# import argparse
# import subprocess

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True)
#     parser.add_argument("--port", type=int, default=8000)
#     parser.add_argument("--max_model_len", type=int, default=4096)
#     parser.add_argument("--task", type=str, default="reward")
#     parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
#     args = parser.parse_args()

#     print(f"🚀 Starting vLLM server for {args.model} on port {args.port}")

#     subprocess.run([
#         "vllm", "serve",
#         args.model,
#         "--port", str(args.port),
#         "--max-model-len", str(args.max_model_len),
#         "--gpu-memory-utilization", str(args.gpu_memory_utilization),
#         "--task", args.task,
#         "--trust-remote-code"
#     ])

# if __name__ == "__main__":
#     main()
import argparse
import torch
from vllm import LLM

# ✅ Global persistent LLM instance
llm = None

def load_llm_on_gpu0(model_name: str, max_model_len: int):
    global llm
    print(f"🚀 Loading {model_name} on GPU 0 for ThinkPRM verification...")
    
    # Force GPU 0 usage
    torch.cuda.set_device(0)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,          # single GPU
        max_model_len=max_model_len,
        trust_remote_code=True
    )
    print("✅ ThinkPRM model loaded and ready on GPU 0!")

def get_llm():
    global llm
    if llm is None:
        raise RuntimeError("❌ LLM is not loaded. Call load_llm_on_gpu0 first!")
    return llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    load_llm_on_gpu0(args.model, args.max_model_len)

    print("✅ Model will persist in GPU memory. You can now make verification calls from PPO (GPU 1).")
    print("Press CTRL+C to exit (this will free GPU 0 memory).")

    # ✅ Keeps the process alive for PPO training to make calls
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("🛑 Exiting and freeing GPU memory.")

if __name__ == "__main__":
    main()
