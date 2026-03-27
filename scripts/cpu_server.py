#!/usr/bin/env python3
"""
Server di inferenza compatibile con OpenAI tramite HuggingFace transformers (CPU).

Fornisce la stessa API /v1/chat/completions di vLLM così il resto del demo
(04_flask_app.py, script openai client) funziona senza modifiche.

Utilizzo:
    python3 scripts/cpu_server.py --model models/Qwen2-1.5B-Ita --port 8000
"""

import argparse
import json
import time
import uuid
import os

import torch
from flask import Flask, Response, jsonify, request, stream_with_context
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

MODEL_PATH: str = ""
MODEL_NAME: str = ""
tokenizer = None
model = None


# ---------------------------------------------------------------------------
# Caricamento del modello
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> None:
    global tokenizer, model, MODEL_NAME
    MODEL_NAME = os.path.basename(os.path.abspath(model_path))
    print(f"Loading tokenizer from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Loading model (CPU, bfloat16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Model ready.", flush=True)


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def build_prompt(messages: list[dict]) -> str:
    """Applica il chat template se il tokenizer ne ha uno, altrimenti usa il fallback."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback semplice
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def generate(prompt: str, max_tokens: int, temperature: float, stop: list[str]) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Applica le sequenze di stop
    for s in (stop or []):
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]

    return text.strip()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/v1/models")
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    body = request.get_json(force=True)
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    stop = body.get("stop", [])

    prompt = build_prompt(messages)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if not stream:
        text = generate(prompt, max_tokens, temperature, stop)
        return jsonify({
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        })

    # --- Streaming (token per token via SSE) ---
    def token_stream():
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        generated = input_ids.clone()

        for _ in range(max_tokens):
            with torch.no_grad():
                out = model(generated)
            logits = out.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=-1)

            token_text = tokenizer.decode(next_id[0], skip_special_tokens=True)

            # Ferma su EOS
            if next_id.item() == tokenizer.eos_token_id:
                break

            # Ferma sulle sequenze di stop (controllo semplice)
            stop_hit = any(token_text == s for s in (stop or []))

            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": "stop" if stop_hit else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            if stop_hit:
                break

        # Chunk finale di completamento
        done_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(token_stream()),
        content_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Punto di ingresso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU inference server (OpenAI-compatible)")
    parser.add_argument("--model", default="models/Qwen2-1.5B-Ita", help="Path to model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model)

    print(f"Starting server on {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=False)
