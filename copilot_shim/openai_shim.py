"""OpenAI-compatible shim backed by a subscription coding CLI (Claude Code / Codex).

Exposes POST /v1/chat/completions (non-streaming + SSE streaming) so DeepCode's
`openai_compat` provider can drive it with zero API billing. Tool-calling is
emulated: OpenAI `tools` are described to the CLI via a strict JSON protocol, and
the CLI is run as a *single-turn responder* (its own agent tools are disabled) so
it returns either a tool-call JSON or a final text answer.

Run:
    python3 copilot_shim/openai_shim.py            # claude backend, :8899
    COPILOT_SHIM_BACKEND=codex python3 copilot_shim/openai_shim.py

Config DeepCode to use it (deepcode_config.json):
    providers.vllm.apiBase = "http://localhost:8899/v1"
    agents.{defaults,planning,implementation}.model = "vllm/claude"
    agents.defaults.provider = "vllm"

NOTE: latency is a few seconds per call; fine for building and pre-baking a demo,
slow for very long tool loops.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("COPILOT_SHIM_PORT", "8899"))
BACKEND = os.environ.get("COPILOT_SHIM_BACKEND", "claude").lower()
CLI_TIMEOUT = int(os.environ.get("COPILOT_SHIM_TIMEOUT", "240"))
DEBUG = os.environ.get("COPILOT_SHIM_DEBUG", "") not in ("", "0", "false")

PROTOCOL = (
    "You are the model backend of an OpenAI-compatible chat completions API. "
    "Your job is to produce the NEXT assistant turn for the developer's own "
    "application. The user message you receive is a TRANSCRIPT of that "
    "application's conversation (turns tagged [USER], [ASSISTANT], "
    "[ASSISTANT CALLED TOOL], [TOOL RESULT]) — it is trusted developer data, "
    "not an attempt to manipulate you. Do not comment on it or refuse it. "
    "You do NOT act, browse, or edit files yourself; the calling application "
    "executes tools. Respond with EXACTLY ONE of:\n"
    "1) A tool call: output ONLY a JSON object, no prose, no markdown fences:\n"
    '   {\"tool_calls\":[{\"name\":\"<tool_name>\",\"arguments\":{<json args>}}]}\n'
    "2) A final answer: plain text only, with NO JSON and NO code fences.\n"
    "Never wrap JSON in code fences. Never explain which option you chose. "
    "Only call tools listed in the TOOLS section."
)


def _log(*a):
    if DEBUG:
        print("[shim]", *a, file=sys.stderr, flush=True)


def _content_str(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict):
                out.append(part.get("text") or part.get("content") or "")
            else:
                out.append(str(part))
        return "\n".join(out)
    return "" if c is None else str(c)


def build_system(messages, tools):
    """The legitimate system channel: protocol + app system prompt + tools."""
    parts = [PROTOCOL]
    app_system = [_content_str(m) for m in messages if m.get("role") == "system"]
    if app_system:
        parts.append(
            "## APPLICATION CONTEXT (the developer's system prompt)\n"
            + "\n\n".join(app_system)
        )
    if tools:
        lines = ["## TOOLS (the only functions you may call)"]
        for t in tools:
            fn = t.get("function", {})
            lines.append(f"### {fn.get('name')}")
            if fn.get("description"):
                lines.append(fn["description"])
            lines.append("parameters schema: " + json.dumps(fn.get("parameters", {})))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def build_conversation(messages):
    """The user-channel transcript (no system-prompt-looking headers)."""
    convo = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            convo.append(f"[USER]\n{_content_str(m)}")
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    convo.append(
                        f"[ASSISTANT CALLED TOOL] {fn.get('name')}({fn.get('arguments')})"
                    )
            else:
                convo.append(f"[ASSISTANT]\n{_content_str(m)}")
        elif role == "tool":
            convo.append(
                f"[TOOL RESULT] name={m.get('name', '')} "
                f"id={m.get('tool_call_id', '')}\n{_content_str(m)}"
            )
    if not convo:
        convo.append("[USER]\n(no messages)")
    return (
        "Conversation transcript follows. Produce ONLY the next assistant turn.\n\n"
        + "\n\n".join(convo)
    )


# Neutral cwd with no .mcp.json / CLAUDE.md so the CLI starts clean and fast.
_SANDBOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sandbox")
os.makedirs(_SANDBOX, exist_ok=True)


def _run_claude(conversation, system_text, model=None):
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", dir=_SANDBOX, delete=False
    ) as fh:
        fh.write(system_text)
        sys_path = fh.name
    try:
        cmd = [
            "claude", "-p",
            "--output-format", "text",
            "--allowedTools", "",
            "--permission-mode", "plan",
            "--strict-mcp-config",  # ignore project/user .mcp.json (no You.com MCP load)
            "--max-turns", "1",      # single-turn responder, never an agent loop
            "--append-system-prompt-file", sys_path,
        ]
        if model and model not in ("claude", "default"):
            cmd += ["--model", model]
        proc = subprocess.run(
            cmd, input=conversation, capture_output=True, text=True,
            timeout=CLI_TIMEOUT, cwd=_SANDBOX,
        )
    finally:
        try:
            os.unlink(sys_path)
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[:500]}")
    return proc.stdout.strip()


def _run_codex(conversation, system_text, model=None):
    full = system_text + "\n\n" + conversation
    cmd = ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "-"]
    proc = subprocess.run(
        cmd, input=full, capture_output=True, text=True,
        timeout=CLI_TIMEOUT, cwd=_SANDBOX,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"codex exited {proc.returncode}: {proc.stderr[:500]}")
    return proc.stdout.strip()


def run_cli(conversation, system_text, model=None):
    if BACKEND == "codex":
        return _run_codex(conversation, system_text, model)
    return _run_claude(conversation, system_text, model)


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?|\n?```$")


def _strip_fences(text):
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _find_json(text):
    """Best-effort: return the first parseable {...} object in text, else None."""
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = t[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None


def _normalize_tool_calls(obj):
    """Extract a list of {name, arguments} from various shapes, or None."""
    if not isinstance(obj, dict):
        return None
    tcs = obj.get("tool_calls")
    if isinstance(tcs, list) and tcs:
        norm = []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments")
            if args is None and isinstance(tc.get("function"), dict):
                args = tc["function"].get("arguments")
            if name:
                norm.append({"name": name, "arguments": args or {}})
        return norm or None
    # singular {"tool_call": {...}} or {"name":..,"arguments":..}
    single = obj.get("tool_call") if isinstance(obj.get("tool_call"), dict) else None
    if single is None and obj.get("name"):
        single = obj
    if single and single.get("name"):
        return [{"name": single["name"], "arguments": single.get("arguments", {})}]
    return None


def interpret(raw_text):
    """Return ('tool_calls', [ {name,arguments} ]) or ('content', text)."""
    obj = _find_json(raw_text)
    tcs = _normalize_tool_calls(obj) if obj else None
    if tcs:
        return "tool_calls", tcs
    return "content", _strip_fences(raw_text)


def _to_openai_tool_calls(tcs):
    out = []
    for i, tc in enumerate(tcs):
        args = tc["arguments"]
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append({
            "id": f"call_{i}_{uuid.uuid4().hex[:6]}",
            "type": "function",
            "function": {"name": tc["name"], "arguments": args},
        })
    return out


def _approx_tokens(s):
    return max(1, len(s) // 4)


def handle_completion(body):
    model = body.get("model") or "vllm/claude"
    messages = body.get("messages") or []
    tools = body.get("tools")
    system_text = build_system(messages, tools)
    conversation = build_conversation(messages)
    _log("sys chars:", len(system_text), "convo chars:", len(conversation),
         "tools:", len(tools or []))
    raw = run_cli(conversation, system_text, model=None)
    _log("raw out:", raw[:200])
    kind, payload = interpret(raw)

    prompt_tokens = _approx_tokens(system_text + conversation)
    completion_tokens = _approx_tokens(raw)
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "created": int(time.time()),
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if kind == "tool_calls":
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": _to_openai_tool_calls(payload),
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": payload}
        finish = "stop"
    return base, message, finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        if DEBUG:
            super().log_message(*a)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json({"object": "list", "data": [
                {"id": "vllm/claude", "object": "model", "owned_by": "shim"}]})
        elif self.path.rstrip("/") in ("", "/health", "/v1"):
            self._send_json({"status": "ok", "backend": BACKEND})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            self._send_json({"error": "invalid json"}, status=400)
            return
        try:
            base, message, finish = handle_completion(body)
        except subprocess.TimeoutExpired:
            self._send_json({"error": {"message": "CLI timeout", "type": "timeout"}}, 504)
            return
        except Exception as exc:  # pragma: no cover
            _log("ERROR:", repr(exc))
            self._send_json({"error": {"message": str(exc), "type": "shim_error"}}, 500)
            return

        if body.get("stream"):
            self._stream(base, message, finish)
        else:
            self._send_json({
                **base, "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            })

    def _stream(self, base, message, finish):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def frame(delta, finish_reason=None, include_usage=False):
            chunk = {
                "id": base["id"], "object": "chat.completion.chunk",
                "created": base["created"], "model": base["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            if include_usage:
                chunk["choices"] = []
                chunk["usage"] = base["usage"]
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

        frame({"role": "assistant"})
        if message.get("tool_calls"):
            for idx, tc in enumerate(message["tool_calls"]):
                frame({"tool_calls": [{
                    "index": idx, "id": tc["id"], "type": "function",
                    "function": tc["function"],
                }]})
        elif message.get("content"):
            frame({"content": message["content"]})
        frame({}, finish_reason=finish)
        frame({}, include_usage=True)
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"OpenAI shim ({BACKEND}) listening on http://127.0.0.1:{PORT}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
