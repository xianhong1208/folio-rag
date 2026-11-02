"""A tiny in-process fake of the OpenAI-compatible model APIs, for adapter tests.

Serves on 127.0.0.1 (ephemeral port) from a daemon thread — no network, no keys:

    POST */embeddings            → deterministic vectors of ``dim`` floats
    POST */chat/completions      → ``chat_reply(body)`` (default: echoes "ok")
    POST */score                 → vLLM score API
    POST */rerank                → Cohere/Jina-style results
    POST */audio/transcriptions  → ``transcribe_reply(fields)`` (default: {"text": "hello"})
    GET  */models                → {"data": [{"id": ...}]}

Every request is recorded in ``server.requests`` as (method, path, headers, body);
``fail_next`` queues status codes (with optional Retry-After) to exercise retries.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple


class FakeModelServer:
    def __init__(self, dim: int = 8, models: Optional[List[str]] = None):
        self.dim = dim
        self.models = models or ["fake-model"]
        self.requests: List[Tuple[str, str, Dict[str, str], Any]] = []
        self.fail_next: List[Tuple[int, Optional[str]]] = []  # (status, retry_after)
        self.chat_reply: Callable[[dict], str] = lambda body: "ok"
        self.transcribe_reply: Callable[[dict], dict] = lambda fields: {"text": "hello"}
        self.rerank_style = "results"  # or "data" (score-style key names)
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, status: int, payload: Any, headers: Optional[dict] = None):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _maybe_fail(self) -> bool:
                with server._lock:
                    if not server.fail_next:
                        return False
                    status, retry_after = server.fail_next.pop(0)
                self._send(status, {"error": {"message": f"injected {status}"}},
                           {"Retry-After": retry_after} if retry_after else None)
                return True

            def do_GET(self):  # noqa: N802
                server.requests.append(("GET", self.path, dict(self.headers), None))
                if self._maybe_fail():
                    return
                if self.path.endswith("/models"):
                    return self._send(200, {"data": [{"id": m} for m in server.models]})
                self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    body = _parse_multipart(raw, ctype)
                else:
                    try:
                        body = json.loads(raw or b"{}")
                    except ValueError:
                        body = raw
                server.requests.append(("POST", self.path, dict(self.headers), body))
                if self._maybe_fail():
                    return
                path = self.path.split("?")[0]
                if path.endswith("/embeddings"):
                    inputs = body.get("input")
                    inputs = [inputs] if isinstance(inputs, str) else inputs
                    data = [{"object": "embedding", "index": i,
                             "embedding": [float((len(str(t)) + j) % 7) / 7 for j in range(server.dim)]}
                            for i, t in enumerate(inputs)]
                    return self._send(200, {"object": "list", "data": data, "model": body.get("model"),
                                            "usage": {"prompt_tokens": 1, "total_tokens": 1}})
                if path.endswith("/chat/completions"):
                    text = server.chat_reply(body)
                    return self._send(200, {
                        "id": "c1", "object": "chat.completion", "created": 0,
                        "model": body.get("model"),
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": text}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
                if path.endswith("/score"):
                    docs = body.get("text_2") or []
                    return self._send(200, {"data": [
                        {"index": i, "score": round(1.0 / (i + 1), 4)} for i in range(len(docs))]})
                if path.endswith("/rerank"):
                    docs = body.get("documents") or body.get("texts") or []
                    items = [{"index": i, "relevance_score": round(1.0 / (i + 1), 4)}
                             for i in range(len(docs))]
                    items.reverse()  # providers return best-first; force re-ordering by index
                    if server.rerank_style == "data":
                        return self._send(200, {"data": [{"index": x["index"], "score": x["relevance_score"]}
                                                         for x in items]})
                    return self._send(200, {"results": items})
                if path.endswith("/audio/transcriptions"):
                    return self._send(200, server.transcribe_reply(body))
                self._send(404, {"error": "not found"})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def posts(self, suffix: str) -> List[Any]:
        return [r[3] for r in self.requests if r[0] == "POST" and r[1].split("?")[0].endswith(suffix)]

    def __enter__(self) -> "FakeModelServer":
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()


def _parse_multipart(raw: bytes, ctype: str) -> dict:
    """Minimal multipart/form-data parser: {field: str | {"filename", "size"}}."""
    boundary = ctype.split("boundary=")[-1].strip().strip('"').encode()
    out: dict = {}
    for part in raw.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        headers = head.decode("latin-1")
        name = headers.split('name="')[1].split('"')[0] if 'name="' in headers else "?"
        if 'filename="' in headers:
            out[name] = {"filename": headers.split('filename="')[1].split('"')[0], "size": len(data)}
        else:
            out[name] = data.decode("utf-8", "replace")
    return out
