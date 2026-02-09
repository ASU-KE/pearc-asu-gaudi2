#!/usr/bin/env python3
"""
CUDA inference benchmark for Llama-3-8B (BF16).
Prints metrics in a format matching optimum-habana's run_generation.py
so the same log-parsing logic works for all three devices.

Uses torch.compile (equivalent of --use_hpu_graphs on Gaudi)
and a manual decode loop for accurate first/rest token latencies.
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = torch.device("cuda")

    print(f"Loading model {args.model_name_or_path} in {dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # ---- torch.compile (equivalent of --use_hpu_graphs on Gaudi) ----
    compile_start = time.perf_counter()
    model = torch.compile(model, mode="default")
    compile_duration = time.perf_counter() - compile_start
    print(f"torch.compile called ({compile_duration:.2f}s), graph will be traced on first run.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- build input batch ----------
    prompt = "DeepSpeed is a machine learning framework."
    inputs = tokenizer(
        [prompt] * args.batch_size,
        return_tensors="pt",
        padding=True,
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = input_ids.shape[1]

    # ---------- manual decode loop for accurate latency breakdown ----------
    torch.cuda.reset_peak_memory_stats()
    new_tokens = 0

    with torch.no_grad():
        # ---- prefill: first forward pass produces first new token ----
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

        torch.cuda.synchronize()
        first_token_s = time.perf_counter() - t_start
        new_tokens += 1

        # update sequences
        generated_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=-1
        )

        # ---- decode: remaining tokens one at a time with KV cache ----
        torch.cuda.synchronize()
        t_decode_start = time.perf_counter()

        for _ in range(args.max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )
            new_tokens += 1

        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t_decode_start

    end2end_s = time.perf_counter() - t_start
    total_tokens = new_tokens * args.batch_size
    throughput = total_tokens / end2end_s
    rest_token_s = decode_s / max(new_tokens - 1, 1)

    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
    mem_max = torch.cuda.max_memory_allocated() / (1024**3)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # ---------- print in the same format optimum-habana uses ----------
    print(f"Throughput (including tokenization) = {throughput:.2f} tokens/s")
    print(f"Average first token latency = {first_token_s * 1000:.2f} ms")
    print(f"Average rest token latency = {rest_token_s * 1000:.2f} ms")
    print(f"Average end to end latency = {end2end_s * 1000:.2f} ms")
    print(f"Memory allocated = {mem_alloc:.2f} GB")
    print(f"Max memory allocated = {mem_max:.2f} GB")
    print(f"Total memory available = {total_mem:.2f} GB")
    print(f"Graph compilation duration = {compile_duration:.2f} s")


if __name__ == "__main__":
    main()
