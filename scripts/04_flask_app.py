#!/usr/bin/env python3
"""
Parte 4 — App Flask Interattiva
Eseguita sul laptop del presentatore. Fa da proxy verso il server vLLM del Jetson
e trasmette le risposte al browser tramite Server-Sent Events (SSE).

Utilizzo:
    VLLM_URL=http://orin:8000 python3 04_flask_app.py
    # oppure con il valore predefinito (http://orin:8000):
    python3 04_flask_app.py
"""

import argparse
import json
import os
import time

import requests
from flask import Flask, Response, render_template_string, request, stream_with_context

VLLM_URL = os.environ.get("VLLM_URL", "http://orin:8000")
MODEL = "/model"
SYSTEM_PROMPT = "Sei un assistente tecnico esperto. Rispondi sempre in italiano."
MAX_TOKENS = 200

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Template HTML incorporato (file singolo, senza build step esterno)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Linux Meeting Palermo 2026 — vLLM Live Demo</title>
  <style>
    :root {
      --green:      #39ff14;
      --green-dim:  #1a7a00;
      --dark:       #0a0a0a;
      --panel:      #0f0f0f;
      --border:     #1f3a1f;
      --text:       #d4f5d4;
      --muted:      #5a8a5a;
      --cursor-col: #39ff14;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--dark);
      color: var(--text);
      font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* sovrapposizione scanline */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background: repeating-linear-gradient(
        to bottom,
        transparent 0px,
        transparent 3px,
        rgba(0,0,0,0.08) 3px,
        rgba(0,0,0,0.08) 4px
      );
      pointer-events: none;
      z-index: 999;
    }

    header {
      background: var(--panel);
      border-bottom: 2px solid var(--green-dim);
      padding: 16px 32px;
      display: flex;
      align-items: flex-start;
      gap: 28px;
    }

    .tux {
      color: var(--green);
      font-size: 0.72rem;
      line-height: 1.25;
      white-space: pre;
      flex-shrink: 0;
      text-shadow: 0 0 8px var(--green);
    }

    .header-text {
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 4px;
    }

    .header-text .event {
      font-size: 0.9rem;
      color: var(--muted);
      letter-spacing: 0.04em;
    }

    .header-text h1 {
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--green);
      text-shadow: 0 0 10px var(--green);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .header-text .subtitle {
      font-size: 0.82rem;
      color: var(--muted);
    }

    .header-text .subtitle em {
      color: var(--text);
      font-style: normal;
    }

    main {
      flex: 1;
      max-width: 960px;
      width: 100%;
      margin: 0 auto;
      padding: 28px 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .prompt-label {
      font-size: 0.78rem;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .prompt-label::before { content: '> '; color: var(--green); }

    textarea {
      width: 100%;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--green);
      font-family: inherit;
      font-size: 1rem;
      padding: 12px 14px;
      resize: vertical;
      min-height: 80px;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
      caret-color: var(--green);
    }
    textarea:focus {
      border-color: var(--green-dim);
      box-shadow: 0 0 8px rgba(57,255,20,0.15);
    }

    .controls-row {
      display: flex;
      align-items: flex-end;
      gap: 24px;
      flex-wrap: wrap;
    }

    .slider-group { flex: 1; min-width: 200px; }

    .slider-label-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 6px;
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .slider-label-row::before { content: '# '; color: var(--green-dim); }

    #temp-value { color: var(--green); }

    input[type="range"] {
      width: 100%;
      accent-color: var(--green);
      cursor: pointer;
    }

    button#send-btn {
      background: transparent;
      color: var(--green);
      border: 2px solid var(--green-dim);
      border-radius: 4px;
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: 700;
      padding: 11px 28px;
      cursor: pointer;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      transition: background 0.15s, box-shadow 0.15s;
      white-space: nowrap;
    }
    button#send-btn:hover {
      background: rgba(57,255,20,0.08);
      box-shadow: 0 0 12px rgba(57,255,20,0.3);
    }
    button#send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

    .response-label {
      font-size: 0.78rem;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .response-label::before { content: '$ '; color: var(--green); }

    #response-box {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 16px;
      min-height: 180px;
      font-family: inherit;
      font-size: 0.95rem;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
    }

    #response-box .placeholder {
      color: var(--muted);
      font-style: italic;
    }

    #cursor {
      display: inline-block;
      width: 8px;
      height: 1em;
      background: var(--cursor-col);
      vertical-align: text-bottom;
      animation: blink 0.75s step-end infinite;
      box-shadow: 0 0 6px var(--green);
    }
    @keyframes blink { 50% { opacity: 0; } }

    #stats-bar {
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      font-size: 0.82rem;
      color: var(--muted);
      min-height: 20px;
      font-family: inherit;
    }
    #stats-bar .lbl { color: var(--muted); }
    #stats-bar .val { color: var(--green); font-weight: 700; text-shadow: 0 0 6px var(--green); }

    footer {
      border-top: 1px solid var(--border);
      text-align: center;
      padding: 12px;
      font-size: 0.73rem;
      color: var(--muted);
    }
    footer a { color: var(--green-dim); text-decoration: none; }

    .error-msg { color: #ff4444; }
  </style>
</head>
<body>

<header>
  <pre class="tux">    .--.
   |o_o |
   |:_/ |
  //   \\ \\
 (|     | )
/'\\_   _/`\\
\\___)=(___/</pre>
  <div class="header-text">
    <h1>Linux Meeting Palermo 2026</h1>
    <span class="event">vLLM on Jetson Orin Nano</span>
    <span class="subtitle">
      model: <em>Qwen2.5-1.5B-GPTQ-Int4</em> &nbsp;|&nbsp;
      hw: <em>NVIDIA Jetson Orin Nano 8GB (aarch64)</em> &nbsp;|&nbsp;
      engine: <em>vLLM + PagedAttention</em>
    </span>
  </div>
</header>

<main>
  <div>
    <div class="prompt-label">prompt</div>
    <textarea id="user-prompt" rows="3"
      placeholder="Scrivi qui la tua domanda in italiano...">Cos'è il kernel Linux?</textarea>
  </div>

  <div class="controls-row">
    <div class="slider-group">
      <div class="slider-label-row">
        temperature &nbsp;<span id="temp-value">0.7</span>
      </div>
      <input type="range" id="temperature" min="0" max="1.5" step="0.05" value="0.7" />
    </div>
    <button id="send-btn" onclick="sendRequest()">[ invio ]</button>
  </div>

  <div>
    <div class="response-label">output</div>
    <div id="response-box">
      <span class="placeholder">_ in attesa di input...</span>
    </div>
  </div>

  <div id="stats-bar"></div>
</main>

<footer>
  Linux Meeting Palermo 2026 &nbsp;|&nbsp;
  Running 100% locally — no cloud, no internet required &nbsp;|&nbsp;
  API: OpenAI-compatible (SSE streaming)
</footer>

<script>
  const tempSlider = document.getElementById('temperature');
  const tempVal    = document.getElementById('temp-value');
  tempSlider.addEventListener('input', () => { tempVal.textContent = tempSlider.value; });

  let currentSource = null;

  function sendRequest() {
    const prompt = document.getElementById('user-prompt').value.trim();
    if (!prompt) return;

    const temperature = parseFloat(tempSlider.value);
    const btn         = document.getElementById('send-btn');
    const box         = document.getElementById('response-box');
    const statsBar    = document.getElementById('stats-bar');

    // Reimposta UI
    if (currentSource) { currentSource.close(); currentSource = null; }
    box.innerHTML = '<span id="cursor"></span>';
    statsBar.innerHTML = '';
    btn.disabled = true;

    let text         = '';
    let tokenCount   = 0;
    let startTime    = null;
    let ttft         = null;
    let firstToken   = true;

    const url = `/stream?prompt=${encodeURIComponent(prompt)}&temperature=${temperature}`;
    const evtSource = new EventSource(url);
    currentSource   = evtSource;

    evtSource.addEventListener('token', (e) => {
      if (startTime === null) startTime = performance.now();
      const token = JSON.parse(e.data);

      if (firstToken) {
        ttft = (performance.now() - startTime) / 1000;
        firstToken = false;
      }

      text += token;
      tokenCount++;

      const cursor = document.getElementById('cursor');
      if (cursor) {
        box.innerHTML = escapeHtml(text) + '<span id="cursor"></span>';
      } else {
        box.innerHTML = escapeHtml(text);
      }
      box.scrollTop = box.scrollHeight;
    });

    evtSource.addEventListener('done', (e) => {
      evtSource.close();
      currentSource = null;
      btn.disabled = false;

      // Rimuove il cursore
      const cursor = document.getElementById('cursor');
      if (cursor) cursor.remove();

      const data  = JSON.parse(e.data);
      const total = data.total_time;
      const toks  = data.completion_tokens || tokenCount;
      const tps   = toks / total;

      statsBar.innerHTML =
        `<span class="lbl">tokens</span> <span class="val">${toks}</span> &nbsp;|&nbsp; ` +
        `<span class="lbl">time</span> <span class="val">${total.toFixed(2)}s</span> &nbsp;|&nbsp; ` +
        `<span class="lbl">TTFT</span> <span class="val">${ttft ? (ttft * 1000).toFixed(0) + 'ms' : '—'}</span> &nbsp;|&nbsp; ` +
        `<span class="lbl">speed</span> <span class="val">${tps.toFixed(1)} tok/s</span>`;
    });

    evtSource.addEventListener('error_msg', (e) => {
      evtSource.close();
      currentSource = null;
      btn.disabled = false;
      const msg = JSON.parse(e.data);
      box.innerHTML = `<span class="error-msg">✗ Error: ${escapeHtml(msg)}</span>`;
    });

    evtSource.onerror = () => {
      evtSource.close();
      currentSource = null;
      btn.disabled = false;
      if (!box.querySelector('.error-msg')) {
        box.innerHTML = '<span class="error-msg">✗ Connection lost. Is the server running?</span>';
      }
    };
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Invio invia il prompt; Shift+Invio inserisce una nuova riga
  document.getElementById('user-prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendRequest();
    }
  });
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Route Flask
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/stream")
def stream():
    prompt = request.args.get("prompt", "")
    temperature = float(request.args.get("temperature", 0.7))

    @stream_with_context
    def generate():
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": temperature,
            "stream": True,
        }

        start = time.perf_counter()
        completion_tokens = 0

        try:
            with requests.post(
                f"{VLLM_URL}/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer none"},
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        completion_tokens += 1
                        yield f"event: token\ndata: {json.dumps(content)}\n\n"

                    # Recupera usage se presente
                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens"):
                        completion_tokens = usage["completion_tokens"]

        except requests.exceptions.ConnectionError as exc:
            yield f"event: error_msg\ndata: {json.dumps(str(exc))}\n\n"
            return
        except requests.exceptions.HTTPError as exc:
            yield f"event: error_msg\ndata: {json.dumps(str(exc))}\n\n"
            return

        total = time.perf_counter() - start
        done_payload = json.dumps({
            "total_time": round(total, 3),
            "completion_tokens": completion_tokens,
        })
        yield f"event: done\ndata: {done_payload}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Punto di ingresso
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Part 4 — Flask web UI for vLLM")
    parser.add_argument(
        "--url",
        default=None,
        help="vLLM server base URL (overrides VLLM_URL env var, default: http://orin:8000)",
    )
    parser.add_argument("--port", type=int, default=5000, help="Flask listen port (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Flask listen host (default: 0.0.0.0)")
    args = parser.parse_args()

    global VLLM_URL
    if args.url:
        VLLM_URL = args.url

    print(f"  vLLM backend : {VLLM_URL}")
    print(f"  Flask UI     : http://localhost:{args.port}")
    print(f"  Open the URL above in your browser, then chat with the model!")
    print()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
