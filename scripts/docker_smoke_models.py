"""Part of scripts/docker_smoke.sh: a stub of the OpenAI-compatible model APIs, run as its own container.

docker_smoke.sh starts it from the Folio image itself (standard library only) as the compose service `models`, and
points every model role of the containerised Folio at it:

    POST */embeddings        deterministic bag-of-words vectors (a text and a query sharing words are similar)
    POST */chat/completions  one fixed sentence (the contextual prefix written at index time)
    POST */rerank, */score   a high score for every document
    GET  */models            the model list

Every request is printed as `STUB <method> <path>`, so the smoke test can check from `docker compose logs models` that
indexing and search really called the models.
"""

import hashlib
import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = int(os.environ.get("STUB_DIMENSION", "256"))
REPLY = "This passage is part of the Folio Docker smoke-test document."


def embed(text: str) -> list:
    vector = [0.0] * DIM
    for word in re.findall(r"\w+", str(text).lower()):
        digest = hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:4], "big") % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        print(f"STUB GET {self.path}", flush=True)
        if self.path.rstrip("/").endswith("/models"):
            return self.send({"object": "list", "data": [{"id": "stub", "object": "model"}]})
        self.send({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        print(f"STUB POST {path}", flush=True)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            body = {}
        if path.endswith("/embeddings"):
            inputs = body.get("input")
            inputs = [inputs] if isinstance(inputs, str) else (inputs or [])
            return self.send({"object": "list", "model": body.get("model"),
                              "data": [{"object": "embedding", "index": i, "embedding": embed(t)}
                                       for i, t in enumerate(inputs)],
                              "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        if path.endswith("/chat/completions"):
            return self.send({"id": "stub", "object": "chat.completion", "created": 0, "model": body.get("model"),
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": REPLY}}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        if path.endswith("/rerank"):
            documents = body.get("documents") or body.get("texts") or []
            return self.send({"results": [{"index": i, "relevance_score": 0.95} for i in range(len(documents))]})
        if path.endswith("/score"):
            documents = body.get("text_2") or []
            documents = [documents] if isinstance(documents, str) else documents
            return self.send({"data": [{"index": i, "score": 0.95} for i in range(len(documents))]})
        self.send({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("STUB_PORT", "8000"))
    print(f"STUB listening on {port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
