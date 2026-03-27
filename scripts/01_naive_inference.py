#!/usr/bin/env python3
"""
Parte 1 — Baseline Naive (HuggingFace Transformers)
Eseguito direttamente sul Jetson Orin Nano con transformers.
Mostra quanto sia lenta model.generate() senza un framework di serving.
"""

import os
import sys
import json
import logging
import warnings
import time
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The following generation flags are not valid")

# Il BFC allocator predefinito di PyTorch emette grandi chiamate cudaMalloc contigue
# (~1 GB per il layer di embedding) che l'IOMMU NvMap del Tegra rifiuta
# (errore 12 = ENOMEM sulla mappatura IOVA contigua).
# cudaMallocAsync usa invece l'allocatore VMM stream-ordered di CUDA, che
# mappa lo spazio di indirizzo virtuale indipendentemente dalle pagine fisiche
# e gestisce correttamente pool di memoria unificata frammentati su Tegra.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

MODEL_ID = str(Path(__file__).parent.parent / "models" / "Qwen2-1.5B-Ita")
STATS_FILE = Path(__file__).parent / ".naive_stats.json"
PROMPTS_FILE = Path(__file__).parent / "PROMPTS"

_prompts = [l.strip() for l in PROMPTS_FILE.read_text().splitlines() if l.strip()]
PROMPT = _prompts[0] if _prompts else "Cos'è il kernel Linux?"

SYSTEM_PROMPT = "Sei un assistente tecnico esperto. Rispondi sempre in italiano."
MAX_NEW_TOKENS = 30


def print_separator():
    print("─" * 62)



def main():
    print_separator()
    print("  Part 1 — Naive Inference Baseline (HuggingFace Transformers)")
    print_separator()
    print(f"  Model  : {MODEL_ID}")
    print(f"  Prompt : {PROMPT}")
    print_separator()

    # Import pesanti rimandati a dopo la stampa dell'intestazione
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **kw: None  # evita OOM IOMMU del Tegra

    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Prova prima CUDA; ricade su CPU se l'IOMMU del Tegra rifiuta le grandi
    # richieste cudaMalloc contigue di PyTorch. vLLM ha un proprio gestore
    # della memoria CUDA che gestisce correttamente questo caso — transformers no.
    # Il fallback è esso stesso un punto del demo: senza vLLM, non si riesce
    # nemmeno a caricare il modello sulla GPU in modo affidabile su hardware
    # con memoria unificata vincolata.
    device = "cpu"
    if torch.cuda.is_available():
        import gc
        from pathlib import Path as _P
        from safetensors import safe_open
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from transformers import AutoConfig

        print("[2/3] Loading model, attempting CUDA...")
        load_start = time.perf_counter()
        try:
            config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
            with init_empty_weights():
                _m = AutoModelForCausalLM.from_config(config)
            _m.tie_weights()
            checkpoint = str(_P(MODEL_ID) / "model.safetensors")
            with safe_open(checkpoint, framework="pt", device="cpu") as f:
                for key in f.keys():
                    cpu_tensor = f.get_tensor(key).to(dtype=torch.float16)
                    set_module_tensor_to_device(_m, key, "cuda:0", value=cpu_tensor)
                    del cpu_tensor
            gc.collect()
            torch.cuda.synchronize()
            model = _m
            device = "cuda"
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            print(f"  [!] CUDA load failed: {type(exc).__name__}")
            print("  [!] Jetson's Tegra IOMMU rejects PyTorch's large cudaMalloc")
            print("  [!] requests. This is exactly the problem vLLM solves —")
            print("  [!] falling back to CPU for this baseline.\n")
            try:
                del _m
            except NameError:
                pass
            torch.cuda.empty_cache()
            gc.collect()
        if device == "cpu":
            load_start = time.perf_counter()
    else:
        load_start = time.perf_counter()
        print(f"[2/3] Loading model on CPU (this may take a moment)...")

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_elapsed = time.perf_counter() - load_start
    print(f"      Model loaded in {load_elapsed:.1f}s  [{device.upper()}]")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    print("[3/3] Running inference...\n")

    print(f"  ~~~~ risposta " + "~" * 46)
    print(f"  Q: {PROMPT}")
    print(f"  A: ", end="", flush=True)

    # Loop di decodifica greedy manuale — davvero token per token, senza thread.
    # Usa la KV-cache così ogni passo elabora solo un nuovo token.
    # Il TTFT viene misurato naturalmente dal primo token.
    generated_ids = inputs["input_ids"]
    past_key_values = None
    output_text = ""
    ttft = None

    if device == "cuda":
        torch.cuda.synchronize()
    gen_start = time.perf_counter()
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            if past_key_values is None:
                out = model(input_ids=generated_ids, use_cache=True)
            else:
                out = model(input_ids=next_token_id,
                            past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

            if i == 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                ttft = time.perf_counter() - gen_start

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Decodifica tutti i nuovi token fino ad ora — gestisce correttamente i confini BPE/UTF-8
            new_text = tokenizer.decode(
                generated_ids[0, input_len:].tolist(), skip_special_tokens=True
            )
            delta = new_text[len(output_text):]
            if delta:
                print(delta, end="", flush=True)
            output_text = new_text

    if device == "cuda":
        torch.cuda.synchronize()
    gen_elapsed = time.perf_counter() - gen_start
    print()  # a capo dopo l'output in streaming

    n = len(tokenizer.encode(output_text, add_special_tokens=False))
    tps = n / gen_elapsed if gen_elapsed > 0 else 0.0
    print()

    print("  ~~~ stats ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"  TTFT        {ttft:>7.2f} s")
    print(f"  wall time   {gen_elapsed:>7.2f} s")
    print(f"  tokens      {n:>7d}")
    print(f"  throughput  {tps:>7.1f} tok/s")
    print("  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()

    stats = {"ttft_ms": ttft, "total_s": gen_elapsed,
             "tokens": n, "tps": tps, "device": device}
    try:
        STATS_FILE.write_text(json.dumps(stats))
    except OSError:
        pass
    print()


if __name__ == "__main__":
    main()
