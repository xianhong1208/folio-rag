"""Per-role health probes: one minimal *real* request per model role, using the saved
configuration (the key never leaves the server and is never echoed).

    embedding    POST /embeddings ["ping"]          → checks the vector size vs rag.embedding.dimension
    *_llm        POST /chat/completions, 16 tokens
    vision       POST /chat/completions with an 8×8 PNG → proves the model accepts images
    rerank       the configured rerank style, 1 document
    asr          POST /audio/transcriptions with 0.5 s of silence
    in-process   no request: reports whether local weights are present (no model is loaded)

Results carry only a status (ok / down / disabled / local / unset), the latency and a
short detail (HTTP status, exception class, vector size) — never response bodies, URLs
with credentials or keys.
"""

from __future__ import annotations

import io
import struct
import time
import wave
import zlib
from typing import Any, Dict, Optional

from src.config.roles import LLM_ROLES, resolve_llm, role_summary

ROLES = ("embedding", *LLM_ROLES, "rerank", "vision", "asr")
PROBE_TIMEOUT = 20.0


def _result(role: str, status: str, detail: str, ms: Optional[int] = None,
            model: Optional[str] = None) -> Dict[str, Any]:
    return {"role": role, "status": status, "detail": detail, "latency_ms": ms, "model": model}


def _err(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"HTTP {status}" if status else type(exc).__name__


def _tiny_png() -> bytes:
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * 8 for _ in range(8))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _silent_wav(seconds: float = 0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


def _client(cfg):
    from src.domain.rag.llm_client import create_openai_client
    update = {"max_retries": 0, "timeout": min(float(getattr(cfg, "timeout", PROBE_TIMEOUT) or PROBE_TIMEOUT),
                                               PROBE_TIMEOUT)}
    return create_openai_client(cfg.model_copy(update=update) if hasattr(cfg, "model_copy") else cfg)


def _probe_embedding(cfg) -> str:
    client = _client(cfg)
    try:
        resp = client.embeddings.create(model=cfg.azure_deployment or cfg.model, input=["ping"])
    finally:
        client.close()
    dim = len(resp.data[0].embedding)
    if dim != cfg.dimension:
        raise _Mismatch(f"vector has {dim} dimensions but rag.embedding.dimension is {cfg.dimension}")
    return f"{dim}-dim vector"


def _probe_chat(cfg, with_image: bool = False) -> str:
    import base64
    content: Any = "Reply with OK."
    if with_image:
        uri = "data:image/png;base64," + base64.b64encode(_tiny_png()).decode("ascii")
        content = [{"type": "text", "text": "Reply with OK."},
                   {"type": "image_url", "image_url": {"url": uri}}]
    client = _client(cfg)
    try:
        resp = client.chat.completions.create(
            model=cfg.azure_deployment or cfg.model,
            messages=[{"role": "user", "content": content}], max_tokens=16, temperature=0)
    finally:
        client.close()
    return f"replied ({getattr(resp.choices[0], 'finish_reason', None) or 'ok'})"


def _probe_rerank(cfg) -> str:
    import httpx
    from src.domain.rag.rerank_api import RerankApi
    api = RerankApi(cfg.api, cfg.base_url, cfg.model, cfg.endpoint_path)
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    with httpx.Client(timeout=min(cfg.timeout, PROBE_TIMEOUT), follow_redirects=False) as c:
        resp = c.post(api.url, json=api.build_payload("ping", ["ping"]), headers=headers)
    resp.raise_for_status()
    return f"score {api.parse_scores(resp.json(), 1)[0]:.3f}"


def _probe_asr(cfg) -> str:
    import httpx
    from src.utils.http_retry import join_endpoint
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    data = {"response_format": "json"}
    if cfg.model:
        data["model"] = cfg.model
    with httpx.Client(timeout=min(cfg.timeout, PROBE_TIMEOUT), follow_redirects=False) as c:
        resp = c.post(join_endpoint(cfg.base_url, "/audio/transcriptions"), data=data,
                      files={"file": ("probe.wav", _silent_wav(), "audio/wav")}, headers=headers)
    resp.raise_for_status()
    return "transcribed 0.5 s of silence"


class _Mismatch(Exception):
    pass


def _asr_ready(rag) -> bool:
    try:
        from src.domain.rag.asr_provider import create_asr_provider
        return bool(create_asr_provider(getattr(rag, "asr", None)).available())
    except Exception:
        return False


def probe_role(rag, role: str) -> Dict[str, Any]:
    """Run the probe for ``role`` against the saved config (blocking; call off the loop)."""
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r} (expected one of {ROLES})")
    summary = {r["role"]: r for r in role_summary(rag)}[role]
    model = summary.get("model")
    if summary["runs"] == "off":
        return _result(role, "disabled", "Disabled in settings", model=model)
    if summary["runs"] == "in-process":
        if role == "asr":
            ready = _asr_ready(rag)
            return _result(role, "local" if ready else "down",
                           "Local weights found" if ready else "Local weights missing under assets/",
                           model=summary.get("provider"))
        return _result(role, "local", "Runs in-process (no endpoint to probe)", model=model)

    t0 = time.perf_counter()
    try:
        if role == "embedding":
            detail = _probe_embedding(rag.embedding)
        elif role in LLM_ROLES:
            detail = _probe_chat(resolve_llm(rag, role))
        elif role == "vision":
            detail = _probe_chat(rag.parsing.vision, with_image=True)
        elif role == "rerank":
            detail = _probe_rerank(rag.rerank)
        else:
            if not rag.asr.base_url:
                return _result(role, "unset", "No endpoint configured", model=model)
            detail = _probe_asr(rag.asr)
        status = "ok"
    except _Mismatch as e:
        status, detail = "down", str(e)
    except Exception as e:  # noqa: BLE001 — only the status code / class name is reported
        status, detail = "down", _err(e)
    return _result(role, status, detail, int((time.perf_counter() - t0) * 1000), model)
