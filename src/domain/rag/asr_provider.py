
"""ASR Provider — abstraction over audio-transcription sources.

Audio transcription moves from a hardcoded docling whisper to a swappable provider (selected via
config `rag.asr`), using an injection pattern: DocumentLoader receives an injected provider rather
than picking an implementation itself.

Implementations:
- DoclingWhisperAsrProvider: wraps docling_convert_once (local whisper turbo, default; no behavior
  change). Returns chunks pre-split by the docling chunker.
- OpenAICompatibleAsrProvider: POST {base_url}/audio/transcriptions (multipart) — OpenAI / Groq /
  self-hosted vLLM whisper / faster-whisper-server all conform to this interface. No pre-split
  chunks (falls back to leaf_splitter's plain-text path).
- FireRedAsrProvider (fireredasr_provider.py): local FireRedASR-AED-L (silencedetect segmentation
  ≤55s / 16k resampling / OpenCC s2twp Simplified→Traditional).

Model-agnostic pre/post-processing (leading-silence trimming, hallucination filtering) does not live
inside a provider — it stays in document_loader's audio dispatch layer (audio_defense).
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import httpx

from src.log import get_api_logger
from src.domain.rag.asr_base import AsrProvider, AsrResult  # noqa: F401(re-export)
from src.domain.rag.docling_loader import docling_convert_once

if TYPE_CHECKING:
    from src.config.model import AsrConfig

logger = get_api_logger()


class DoclingWhisperAsrProvider(AsrProvider):
    """Local docling whisper (default) — wraps docling_convert_once, no behavior change."""

    def available(self) -> bool:
        # Keeps the legacy auto-discover semantics: serviceable only if assets/whisper_models/*.pt exists
        from src.domain.rag.docling_loader import _resolve_whisper_path
        return _resolve_whisper_path() is not None

    def transcribe(
        self, *, audio_path: str, file_name: str,
        max_tokens: Optional[int] = None, tokenizer: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> AsrResult:
        # Passing max_tokens=None through would override convert_once's default of 512 and cause a
        # None//2 TypeError inside refine; the production path always supplies a value.
        full_text, chunk_records = docling_convert_once(
            file_path=audio_path, file_name=file_name,
            max_tokens=max_tokens if max_tokens is not None else 512,
            tokenizer=tokenizer,
            progress_cb=progress_cb,
        )
        # convert_once returns chunk records; audio has no heading/page number, so take plain text
        chunks = [r["text"] if isinstance(r, dict) else r for r in chunk_records]
        return AsrResult(text=full_text, chunks=chunks)


class OpenAICompatibleAsrProvider(AsrProvider):
    """Cloud/self-hosted HTTP ASR: POST {base_url}/audio/transcriptions (multipart)."""

    def __init__(
        self, *, base_url: str, api_key: Optional[str] = None,
        model: Optional[str] = None, request_timeout: float = 300.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = request_timeout

    def transcribe(
        self, *, audio_path: str, file_name: str,
        max_tokens: Optional[int] = None, tokenizer: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> AsrResult:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = {}
        if self._model:
            data["model"] = self._model

        url = f"{self._base_url}/audio/transcriptions"
        logger.info(f"[ASR] cloud transcribe: {file_name} → {url}")
        with open(audio_path, "rb") as fh:
            with httpx.Client(headers=headers, timeout=self._timeout) as client:
                resp = client.post(url, data=data, files={"file": (file_name, fh)})
        resp.raise_for_status()
        text = resp.json().get("text", "")
        return AsrResult(text=text, chunks=None)


def create_asr_provider(asr_cfg: "Optional[AsrConfig]") -> AsrProvider:
    """Factory: build a provider based on config `rag.asr.provider`.

    - None (config unset) → default docling-whisper (backward compatible)
    - openai-compatible missing base_url → immediate ValueError (fails at startup rather than
      surfacing only on the first transcription)
    - unknown provider → ValueError listing the available values
    """
    if asr_cfg is None:
        return DoclingWhisperAsrProvider()

    provider = asr_cfg.provider
    if provider == "docling-whisper":
        return DoclingWhisperAsrProvider()
    if provider == "openai-compatible":
        if not asr_cfg.base_url:
            raise ValueError(
                "rag.asr.provider=openai-compatible requires base_url"
                " (e.g. http://host:8000/v1)"
            )
        return OpenAICompatibleAsrProvider(
            base_url=asr_cfg.base_url,
            api_key=asr_cfg.api_key,
            model=asr_cfg.model,
        )
    if provider == "fireredasr":
        # lazy import: the FireRedASR chain (torch/kaldi) is loaded only when selected
        from src.domain.rag.fireredasr_provider import FireRedAsrProvider
        return FireRedAsrProvider()
    raise ValueError(
        f"Unknown rag.asr.provider: {provider!r} — "
        "available: docling-whisper / openai-compatible / fireredasr"
    )
