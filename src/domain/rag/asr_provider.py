
"""ASR Provider — abstraction over audio-transcription sources.

Audio transcription moves from a hardcoded docling whisper to a swappable provider (selected via
config `rag.asr`), using an injection pattern: DocumentLoader receives an injected provider rather
than picking an implementation itself.

Implementations:
- DoclingWhisperAsrProvider: wraps docling_convert_once (local whisper turbo, default; no behavior
  change). Returns chunks pre-split by the docling chunker.
- OpenAICompatibleAsrProvider: POST {base_url}/audio/transcriptions (multipart) — OpenAI / Groq /
  self-hosted vLLM whisper / faster-whisper-server all conform to this interface. Long audio is
  resampled to 16 kHz mono and uploaded in silence-aligned segments (one POST of a long
  recording exceeds the 25 MB upload limit), each with HTTP retries plus a content guard that
  retries empty-despite-speech and runaway (looping) transcripts. No pre-split chunks
  (falls back to leaf_splitter's plain-text path).
- FireRedAsrProvider (fireredasr_provider.py): local FireRedASR-AED-L (silencedetect segmentation
  ≤55s / 16k resampling / OpenCC s2twp Simplified→Traditional).

Model-agnostic pre/post-processing (leading-silence trimming, hallucination filtering) does not live
inside a provider — it stays in document_loader's audio dispatch layer (audio_defense).
"""

from __future__ import annotations

import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

import httpx

from src.log import get_api_logger
from src.domain.rag.asr_base import AsrProvider, AsrResult  # noqa: F401  (re-export)
from src.domain.rag.docling_loader import docling_convert_once
from src.utils.http_retry import RetryPolicy, join_endpoint, request_with_retry

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
        from src.utils.deprecation import warn_once
        warn_once("asr")  # the built-in speech-to-text goes in 1.5 (logged once per process)
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


# Runaway-generation guard. Fast speech is ~15 chars/s in CJK and ~20 in Latin
# scripts; anything well beyond that, or text that compresses like a loop, is the model
# repeating itself, so the segment is retried rather than indexed.
MAX_CHARS_PER_SECOND = 40.0
MAX_COMPRESSION_RATIO = 4.0  # zlib ratio; whisper treats > 2.4 as suspect, loops reach 10+
# A segment that is at least this much speech must not come back empty.
MIN_SPEECH_RATIO_FOR_TEXT = 0.2
# Segments uploaded concurrently per file (bounded: providers rate-limit per key).
SEGMENT_CONCURRENCY = 2
# Upload the original file untouched when ffmpeg is unavailable and it is small enough.
MAX_SINGLE_UPLOAD_BYTES = 24 * 1024 * 1024


class TranscriptRejected(RuntimeError):
    """A segment's transcript failed the content guard on every attempt."""


def transcript_problem(text: str, seconds: float, speech: float) -> Optional[str]:
    """Why a segment transcript is unacceptable, or None when it looks sane."""
    if not text:
        return "empty transcript for a segment with speech" if speech >= MIN_SPEECH_RATIO_FOR_TEXT else None
    limit = max(200.0, seconds * MAX_CHARS_PER_SECOND)
    if len(text) > limit:
        return f"runaway transcript ({len(text)} chars for {seconds:.0f}s of audio)"
    raw = text.encode("utf-8")
    if len(raw) > 400 and len(raw) / max(1, len(zlib.compress(raw))) > MAX_COMPRESSION_RATIO:
        return "repetitive transcript (generation loop)"
    return None


def _fmt_ts(seconds: float) -> str:
    from src.domain.rag.audio_segments import fmt_timestamp

    return fmt_timestamp(seconds)


class OpenAICompatibleAsrProvider(AsrProvider):
    """Cloud/self-hosted HTTP ASR: POST {base_url}/audio/transcriptions (multipart), segmented."""

    def __init__(
        self, *, base_url: str, api_key: Optional[str] = None,
        model: Optional[str] = None, request_timeout: float = 300.0,
        max_retries: int = 3, segment_seconds: float = 600.0,
        language: Optional[str] = None,
        sleep: Optional[Callable[[float], object]] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._segment_seconds = segment_seconds
        self._language = language
        self._policy = RetryPolicy(max_retries=max_retries, backoff_base=1.0, backoff_max=30.0)
        self._sleep = sleep or time.sleep

    def available(self) -> bool:
        return bool(self._base_url)

    @property
    def url(self) -> str:
        return join_endpoint(self._base_url, "/audio/transcriptions")

    def _prepare(self, audio_path: str, tdir: Path) -> List[Tuple[Path, float, float, float]]:
        """Resample + plan + slice → [(path, start, end, speech_ratio)].

        Without ffmpeg the original file is uploaded whole when it fits one request;
        otherwise the error says what to install.
        """
        from src.domain.rag.audio_segments import (
            _detect_silences, _resample_to_wav16k, _slice_wav, plan_segments,
            speech_ratio, wav_duration,
        )
        wav = tdir / "input16k.wav"
        try:
            _resample_to_wav16k(audio_path, wav)
        except FileNotFoundError:
            size = Path(audio_path).stat().st_size
            if size > MAX_SINGLE_UPLOAD_BYTES:
                raise RuntimeError(
                    "ffmpeg is required to split long audio for cloud transcription "
                    f"({size / 1e6:.0f} MB exceeds a single upload) — install ffmpeg")
            logger.warning("[ASR] ffmpeg not found — uploading the original file in one request")
            return [(Path(audio_path), 0.0, 0.0, 1.0)]
        duration = wav_duration(wav)
        silences = _detect_silences(wav) if duration > self._segment_seconds else []
        segments = plan_segments(duration, silences, max_seg=self._segment_seconds)
        paths = _slice_wav(wav, segments, tdir) if len(segments) > 1 else [wav]
        return [(p, s, e, speech_ratio(s, e, silences)) for p, (s, e) in zip(paths, segments)]

    def _transcribe_segment(self, client: httpx.Client, path: Path, start: float, end: float,
                            speech: float, upload_name: str) -> str:
        data = {"response_format": "json", "temperature": "0"}
        if self._model:
            data["model"] = self._model
        if self._language:
            data["language"] = self._language
        payload = path.read_bytes()  # bytes, not a handle: retries re-send the body
        seconds = max(end - start, 0.0) or len(payload) / 32000.0
        problem = None
        for attempt in range(self._max_retries + 1):
            resp = request_with_retry(
                client, "POST", self.url, policy=self._policy, sleep=self._sleep,
                data=data, files={"file": (upload_name, payload)},
            )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            problem = transcript_problem(text, seconds, speech)
            if problem is None:
                return text
            logger.warning(f"[ASR] {upload_name} {_fmt_ts(start)}–{_fmt_ts(end)}: {problem} "
                           f"(attempt {attempt + 1}/{self._max_retries + 1})")
        raise TranscriptRejected(
            f"segment {_fmt_ts(start)}–{_fmt_ts(end)}: {problem} after {self._max_retries + 1} attempts")

    def transcribe(
        self, *, audio_path: str, file_name: str,
        max_tokens: Optional[int] = None, tokenizer: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> AsrResult:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        with tempfile.TemporaryDirectory(prefix="cloud_asr_") as td:
            segments = self._prepare(audio_path, Path(td))
            total = len(segments)
            logger.info(f"[ASR] cloud transcribe: {file_name} → {self.url} ({total} segment(s))")
            done = 0

            with httpx.Client(headers=headers, timeout=self._timeout) as client:
                def run(i_seg):
                    i, (path, start, end, speech) = i_seg
                    name = file_name if total == 1 else f"{Path(file_name).stem}_{i:04d}.wav"
                    return self._transcribe_segment(client, path, start, end, speech, name)

                texts: List[str] = []
                with ThreadPoolExecutor(max_workers=min(SEGMENT_CONCURRENCY, total)) as pool:
                    for text in pool.map(run, enumerate(segments)):  # input order preserved
                        texts.append(text)
                        done += 1
                        if progress_cb is not None:
                            try:
                                progress_cb("loading", done, total)
                            except Exception:
                                pass
        from src.domain.rag.audio_segments import time_block_records

        records = time_block_records(
            [(start, end, text) for (_, start, end, _), text in zip(segments, texts)])
        return AsrResult(text="\n".join(t for t in texts if t), chunks=records)


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
            request_timeout=float(getattr(asr_cfg, "timeout", 300.0)),
            max_retries=int(getattr(asr_cfg, "max_retries", 3)),
            segment_seconds=float(getattr(asr_cfg, "segment_seconds", 600.0)),
            language=getattr(asr_cfg, "language", None),
        )
    if provider == "fireredasr":
        # lazy import: the FireRedASR chain (torch/kaldi) is loaded only when selected
        from src.domain.rag.fireredasr_provider import FireRedAsrProvider
        return FireRedAsrProvider()
    raise ValueError(
        f"Unknown rag.asr.provider: {provider!r} — "
        "available: docling-whisper / openai-compatible / fireredasr"
    )
