#!/usr/bin/env python3
"""
Parte 2 — Serving vLLM (Singola Richiesta)
Invia lo stesso prompt al server vLLM in esecuzione e confronta le prestazioni
con la baseline naive della Parte 1.
"""

import argparse
import json
import time
import openai
from pathlib import Path

STATS_FILE = Path(__file__).parent / ".naive_stats.json"
NAIVE_FALLBACK = {"ttft_ms": 9.0, "total_s": 35.0, "tokens": 70.0, "tps": 2.0}
PROMPTS_FILE = Path(__file__).parent / "PROMPTS"

PROMPTS = [l.strip() for l in PROMPTS_FILE.read_text().splitlines() if l.strip()][:5]
if len(PROMPTS) < 5:
    _defaults = [
        "Cos'è il kernel Linux?",
        "Spiega il concetto di container in informatica.",
        "Quali sono i vantaggi del software open source?",
        "Cos'è Kubernetes in parole semplici?",
        "Descrivi brevemente cos'è Red Hat.",
    ]
    PROMPTS += _defaults[len(PROMPTS):]
SYSTEM_PROMPT = "Sei un assistente tecnico esperto. Rispondi sempre in italiano."
MODEL = "/model"
MAX_TOKENS = 200

COST_TABLE = """
┌─────────────────────────────────────────────────────────────┐
│              💰 Cost Comparison: Edge vs Cloud               │
├─────────────────────┬──────────────┬────────────────────────┤
│                     │ Jetson Orin  │ Cloud GPU (A10G)       │
│                     │ Nano 8GB     │ AWS g5.xlarge          │
├─────────────────────┼──────────────┼────────────────────────┤
│ Hardware cost       │ ~€250 once   │ —                      │
│ Monthly cost (24/7) │ ~€3 electric │ ~€800/month            │
│ Privacy             │ 100% local   │ Hehehe...              │
│ Latency             │ <50ms local  │ Network dependent      │
│ Internet required   │ No           │ Yes                    │
└─────────────────────┴──────────────┴────────────────────────┘
"""


def print_separator():
    print("─" * 62)


def run_streaming_request(client: openai.OpenAI, prompt: str) -> tuple[float, float, int, str]:
    """Returns (ttft_seconds, total_seconds, num_tokens, text). Prints tokens live."""
    collected_text = ""
    ttft: float | None = None
    completion_tokens = 0

    start = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.7,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            if ttft is None:
                ttft = time.perf_counter() - start
            collected_text += token
            print(token, end="", flush=True)
        if chunk.usage and chunk.usage.completion_tokens:
            completion_tokens = chunk.usage.completion_tokens

    total = time.perf_counter() - start
    if ttft is None:
        ttft = total

    # Fallback: conta con split() se usage non è stato restituito
    if completion_tokens == 0:
        completion_tokens = len(collected_text.split())

    return ttft, total, completion_tokens, collected_text


def main():
    parser = argparse.ArgumentParser(description="Part 2 — vLLM single request")
    parser.add_argument(
        "--url",
        default="http://orin:8000",
        help="vLLM server base URL (default: http://orin:8000)",
    )
    args = parser.parse_args()

    client = openai.OpenAI(base_url=f"{args.url}/v1", api_key="none")

    print_separator()
    print("  Part 2 — vLLM Serving (Single Request)")
    print_separator()
    print(f"  Server  : {args.url}")
    print(f"  Model   : {MODEL}")
    print(f"  Prompts : {len(PROMPTS)} different questions")
    print_separator()

    # Controllo di stato
    try:
        models = client.models.list()
        print(f"  ✓ Connected — available model: {models.data[0].id}")
    except Exception as exc:
        print(f"\n  ✗ Cannot reach vLLM server at {args.url}")
        print(f"    Error: {exc}")
        print("    Make sure vLLM is running and the --url is correct.")
        raise SystemExit(1)

    NUM_RUNS = len(PROMPTS)
    print(f"\n  Running {NUM_RUNS} requests...\n")

    ttfts, totals, token_counts = [], [], []
    run_lines = []
    wall_start = time.perf_counter()

    for i, prompt in enumerate(PROMPTS):
        print(f"  ~~~~ request {i + 1}/{NUM_RUNS} " + "~" * 42)
        print(f"  Q: {prompt}")
        print(f"  A: ", end="", flush=True)
        try:
            ttft, total, n_tok, text = run_streaming_request(client, prompt)
        except Exception as exc:
            print(f"\n  ✗ Request failed: {exc}")
            raise SystemExit(1)
        print()  # newline after streamed tokens
        print()

        ttfts.append(ttft)
        totals.append(total)
        token_counts.append(n_tok)

        tps = n_tok / total if total > 0 else 0.0
        run_lines.append(f"  [{i + 1}/{NUM_RUNS}] {n_tok} tok  {ttft:.2f}s TTFT  {tps:.1f} tok/s")

    avg_ttft = sum(ttfts) / NUM_RUNS
    avg_total = sum(totals) / NUM_RUNS
    avg_tokens = sum(token_counts) / NUM_RUNS
    avg_tps = avg_tokens / avg_total if avg_total > 0 else 0.0
    wall_elapsed = time.perf_counter() - wall_start

    print("  ~~~~ per-request summary " + "~" * 33)
    for line in run_lines:
        print(line)
    print()

    print(f"  ~~~ stats (avg of {NUM_RUNS} runs) ~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"  wall time   {wall_elapsed:>7.2f} s  (all {NUM_RUNS} sequential)")
    print(f"  TTFT        {avg_ttft:>7.2f} s  (avg)")
    print(f"  total time  {avg_total:>7.2f} s  (avg / req)")
    print(f"  tokens      {avg_tokens:>7.0f}  (avg / req)")
    print(f"  throughput  {avg_tps:>7.1f} tok/s  (avg / req)")
    print("  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()
    print("  ~~~ naive vs vllm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if STATS_FILE.exists():
        try:
            naive = json.loads(STATS_FILE.read_text())
            print(f"  (naive stats from {STATS_FILE.name})")
        except Exception:
            naive = NAIVE_FALLBACK
            print("  (naive stats: fallback defaults)")
    else:
        naive = NAIVE_FALLBACK
        print("  (naive stats: fallback defaults -- run 01_naive_inference.py first)")
    naive_device = naive.get("device", "cpu").upper()
    print(f"                 naive ({naive_device})       vllm (GPU)")
    print(f"  TTFT       {naive['ttft_ms']:>12.2f} s   {avg_ttft:>10.2f} s")
    print(f"  throughput {naive['tps']:>12.1f} tok/s{avg_tps:>10.1f} tok/s")
    print( "  concurrent          1 req          unlimited")
    print( "  kv-cache               no               yes")
    print("  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(COST_TABLE)


if __name__ == "__main__":
    main()
