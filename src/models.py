"""Model registry. need = GB of GPU memory held for the engine (weights + KV + overhead); kv = KV-cache GB."""
MODELS = {
    "qwen2.5-0.5b": dict(hf="Qwen/Qwen2.5-0.5B-Instruct-AWQ", params=0.5, family="qwen2.5", need=3, kv=1.2),
    "qwen2.5-1.5b": dict(hf="Qwen/Qwen2.5-1.5B-Instruct-AWQ", params=1.5, family="qwen2.5", need=3.5, kv=1.2),
    "qwen2.5-3b":   dict(hf="Qwen/Qwen2.5-3B-Instruct-AWQ", params=3.1, family="qwen2.5", need=4.2, kv=1.0),
    "qwen2.5-7b":   dict(hf="Qwen/Qwen2.5-7B-Instruct-AWQ", params=7.6, family="qwen2.5", need=8, kv=1.4),
    "llama3.2-1b":  dict(hf="unsloth/Llama-3.2-1B-Instruct", params=1.2, family="llama3", need=4.2, kv=1.0),
    "llama3.2-3b":  dict(hf="AMead10/Llama-3.2-3B-Instruct-AWQ", params=3.2, family="llama3", need=4.4, kv=1.0),
    "llama3.1-8b":  dict(hf="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", params=8.0, family="llama3",
                         need=8.5, kv=1.4),
    # served by the host's long-lived vLLM server (bf16), see src/gen_server.py
    "qwen3-8b":     dict(hf="Qwen/Qwen3-8B", params=8.2, family="qwen3", server=True),
    # robustness variants: same checkpoints, numbered-list system prompt (data/prompts_numbered.jsonl)
    "qwen2.5-1.5b#num": dict(hf="Qwen/Qwen2.5-1.5B-Instruct-AWQ", params=1.5, family="qwen2.5", need=3.3, kv=1.0, prompts="prompts_numbered.jsonl", variant=True),
    "qwen2.5-3b#num": dict(hf="Qwen/Qwen2.5-3B-Instruct-AWQ", params=3.1, family="qwen2.5", need=3.5, kv=0.6, prompts="prompts_numbered.jsonl", variant=True),
    "llama3.2-3b#num": dict(hf="AMead10/Llama-3.2-3B-Instruct-AWQ", params=3.2, family="llama3", need=3.5, kv=0.6, prompts="prompts_numbered.jsonl", variant=True),
    "qwen3-8b#num": dict(hf="Qwen/Qwen3-8B", params=8.2, family="qwen3", server=True, prompts="prompts_numbered.jsonl", variant=True),
}
