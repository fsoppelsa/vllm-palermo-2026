#!/usr/bin/env python3
"""
Parte 3 — Batching vLLM (5 Richieste Parallele)
Dimostra il continuous batching di vLLM: 5 richieste concorrenti servite
in modo efficiente, non sequenzialmente.

IMPORTANTE: Riavviare vLLM con --max-num-seqs 8 prima di eseguire questo script:
    docker stop vllm-qwen-it && docker rm vllm-qwen-it
    # poi eseguire di nuovo docker con --max-num-seqs 8
"""

import argparse
import asyncio
import time
from pathlib import Path

import aiohttp

MODEL = "/model"
SYSTEM_PROMPT = (
    "Sei un assistente tecnico esperto. Rispondi sempre in italiano."
)
MAX_TOKENS = 200
PROMPTS_FILE = Path(__file__).resolve().parent / "PROMPTS"

_prompts = [l.strip() for l in PROMPTS_FILE.read_text().splitlines() if l.strip()][:5]
_defaults = [
    "Cos'è il kernel Linux?",
    "Spiega il concetto di container in informatica.",
    "Quali sono i vantaggi del software open source?",
    "Cos'è Kubernetes in parole semplici?",
    "Descrivi brevemente cos'è Red Hat.",
]
PROMPTS = (_prompts + _defaults[len(_prompts):])[:5]


def print_separator():
    print("─" * 70)


async def single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    idx: int,
    prompt: str,
) -> dict:
    """Esegue una singola richiesta chat-completion e restituisce tempi e statistiche di utilizzo."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": False,
    }

    start = time.perf_counter()
    async with session.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer none"},
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    elapsed = time.perf_counter() - start

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(content.split()))
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "idx": idx,
        "prompt": prompt,
        "content": content,
        "elapsed": elapsed,
        "tokens": completion_tokens,
        "tps": tps,
    }


async def run_all(base_url: str) -> list[dict]:
    """Esegue tutte e 5 le richieste contemporaneamente; stampa ciascuna al completamento."""
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = {
            asyncio.ensure_future(single_request(session, base_url, i + 1, prompt)): i
            for i, prompt in enumerate(PROMPTS)
        }

        print(f"  🚀 Fired {len(tasks)} requests simultaneously — waiting for completions...\n")

        results = []
        for coro in asyncio.as_completed(tasks):
            r = await coro
            if isinstance(r, Exception):
                raise r
            results.append(r)
            print(f"  ✓ Request {r['idx']} done — {r['tokens']} tok in {r['elapsed']:.2f}s  ({r['tps']:.1f} tok/s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Part 3 — vLLM batching (5 parallel)")
    parser.add_argument(
        "--url",
        default="http://orin:8000",
        help="vLLM server base URL (default: http://orin:8000)",
    )
    args = parser.parse_args()

    print_separator()
    print("  Part 3 — vLLM Continuous Batching (5 Parallel Requests)")
    print_separator()
    print(f"  Server : {args.url}")
    print(f"  Model  : {MODEL}")
    print(f"  Sending {len(PROMPTS)} requests SIMULTANEOUSLY via asyncio.gather()")
    print_separator()

    wall_start = time.perf_counter()
    try:
        results = asyncio.run(run_all(args.url))
    except aiohttp.ClientConnectorError as exc:
        print(f"  ✗ Cannot reach vLLM server at {args.url}")
        print(f"    Error: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"  ✗ Request failed: {exc}")
        raise SystemExit(1)
    wall_elapsed = time.perf_counter() - wall_start

    total_tokens = sum(r["tokens"] for r in results)
    sequential_time = sum(r["elapsed"] for r in results)
    effective_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0.0

    print()
    for r in sorted(results, key=lambda x: x["idx"]):
        print(f"  ~~~~ request {r['idx']}/{len(results)} " + "~" * 42)
        print(f"  Q: {r['prompt']}")
        print(f"  A: {r['content'].strip()}")
        print()

    avg_elapsed = sum(r["elapsed"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    speedup = sequential_time / wall_elapsed if wall_elapsed > 0 else 0.0

    print("  ~~~~ per-request summary (non-streaming: no TTFT) " + "~" * 9)
    for r in sorted(results, key=lambda x: x["idx"]):
        print(f"  [{r['idx']}/{len(results)}] {r['tokens']:>4} tok  latency {r['elapsed']:.2f}s  {r['tps']:.1f} tok/s")
    print("  ~~~ stats (5 parallel reqs) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"  wall time   {wall_elapsed:>7.2f} s  ← all 5 finished in this time")
    print(f"  seq. time   {sequential_time:>7.2f} s  ← what 5 serial reqs would cost")
    print(f"  speedup     {speedup:>7.1f} x")
    print(f"  avg latency {avg_elapsed:>7.2f} s / req")
    print(f"  avg tokens  {avg_tokens:>7.0f} / req")
    print(f"  avg tps     {avg_tps:>7.1f} tok/s / req")
    print(f"  eff. tput   {effective_tps:>7.1f} tok/s  (aggregate across all reqs)")
    print("  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()


if __name__ == "__main__":
    main()
