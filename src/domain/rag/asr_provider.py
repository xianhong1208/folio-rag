
"""ASR Provider — audio transcription through an OpenAI-compatible endpoint.

Folio 1.5 has no in-process speech-to-text: audio goes to ``POST {base_url}/audio/transcriptions``
(multipart), which OpenAI, Forgeyard, speaches, vLLM and faster-whisper-server all serve. The
provider is injected into DocumentLoader (``create_asr_provider`` builds it from ``rag.asr``).

- OpenAICompatibleAsrProvider: long audio is resampled to 16 kHz mono and uploaded in
  silence-aligned segments (one POST of a long recording exceeds the 25 MB upload limit), each
  with HTTP retries plus a content guard that retries empty-despite-speech and runaway (looping)
  transcripts. The transcript's Chinese script is converted per ``rag.asr.chinese_conversion``
  (whisper writes Simplified Chinese; the default ``s2twp`` gives Traditional with Taiwan phrasing).
- UnconfiguredAsrProvider: ``rag.asr`` has no base_url. Audio uploads are refused up front
  (``asr_upload_refusal``) and indexing an audio file already stored fails with the same message.
- DisabledAsrProvider: ``rag.asr.enabled: false``.

The providers removed in 1.5 (``docling-whisper``, ``fireredasr``) are refused when the
configuration is loaded (``AsrConfig``), with a pointer to docs/migrate-1.5.md.

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

from src.config.model import ASR_MIGRATION_GUIDE, REMOVED_ASR_PROVIDERS, removed_asr_provider_message
from src.log import get_api_logger
from src.domain.rag.asr_base import AsrProvider, AsrResult  # noqa: F401  (re-export)
from src.utils.http_retry import RetryPolicy, join_endpoint, request_with_retry
from src.utils.metered_http import metered_client

if TYPE_CHECKING:
    from src.config.model import AsrConfig  # noqa: F401

logger = get_api_logger()

MIGRATION_GUIDE = ASR_MIGRATION_GUIDE
NOT_CONFIGURED = (
    "Speech-to-text is not configured: set rag.asr.base_url (env ASR_BASE_URL) to an "
    "OpenAI-compatible transcription endpoint, e.g. Forgeyard's /v1 (see " + MIGRATION_GUIDE + ")")


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


def tls_verify(ca_file: Optional[str]):
    """httpx ``verify`` for the endpoint: the default bundle (httpx honours SSL_CERT_FILE),
    plus ``rag.asr.ca_file`` when set, for a server with its own CA such as Forgeyard's
    (src/utils/tls.py; a missing or unreadable file is an error)."""
    from src.utils.tls import tls_verify as _verify
    return _verify(ca_file, "rag.asr.ca_file")


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
        chinese_conversion: Optional[str] = "s2twp",
        ca_file: Optional[str] = None,
        sleep: Optional[Callable[[float], object]] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._segment_seconds = segment_seconds
        self._language = language
        self._chinese_conversion = chinese_conversion
        self._ca_file = ca_file
        self._policy = RetryPolicy(max_retries=max_retries, backoff_base=1.0, backoff_max=30.0)
        self._sleep = sleep or time.sleep

    def available(self) -> bool:
        return bool(self._base_url)

    def _verify(self):
        return tls_verify(self._ca_file)

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
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"The speech-to-text endpoint refused the request (HTTP {resp.status_code}): "
                    "check rag.asr.api_key (ASR_API_KEY) and that the key may use rag.asr.model")
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

            with metered_client("asr", headers=headers, timeout=self._timeout,
                                verify=self._verify()) as client:
                def run(i_seg):
                    i, (path, start, end, speech) = i_seg
                    name = file_name if total == 1 else f"{Path(file_name).stem}_{i:04d}.wav"
                    return self._transcribe_segment(client, path, start, end, speech, name)

                texts: List[str] = []
                try:
                    with ThreadPoolExecutor(max_workers=min(SEGMENT_CONCURRENCY, total)) as pool:
                        for text in pool.map(run, enumerate(segments)):  # input order preserved
                            texts.append(text)
                            done += 1
                            if progress_cb is not None:
                                try:
                                    progress_cb("loading", done, total)
                                except Exception:
                                    pass
                except httpx.TransportError as e:
                    # After the retries: say which endpoint, not just "ConnectError".
                    if "CERTIFICATE_VERIFY_FAILED" in str(e):
                        raise RuntimeError(
                            f"The speech-to-text endpoint {self.url} has a TLS certificate this server "
                            "does not trust: set rag.asr.ca_file (ASR_CA_FILE) to its CA, e.g. "
                            "Forgeyard's own CA") from e
                    raise RuntimeError(
                        f"The speech-to-text endpoint {self.url} could not be reached "
                        f"({type(e).__name__}): check rag.asr.base_url and that the server is up") from e
        from src.domain.rag.audio_segments import time_block_records
        from src.domain.rag.han_script import convert_chinese

        texts = [convert_chinese(t, self._chinese_conversion) for t in texts]
        records = time_block_records(
            [(start, end, text) for (_, start, end, _), text in zip(segments, texts)])
        return AsrResult(text="\n".join(t for t in texts if t), chunks=records)


class DisabledAsrProvider(AsrProvider):
    """``rag.asr.enabled: false``: nothing is transcribed, so an incomplete endpoint is not an error."""

    def available(self) -> bool:
        return False

    def transcribe(self, *, audio_path: str, file_name: str, max_tokens: Optional[int] = None,
                   tokenizer: Optional[str] = None, progress_cb: Optional[Callable] = None) -> AsrResult:
        raise RuntimeError(f"Speech-to-text is disabled (rag.asr.enabled=false): '{file_name}' was not transcribed")


class UnconfiguredAsrProvider(AsrProvider):
    """``rag.asr`` is enabled but has no base_url: the service starts, audio is refused."""

    def available(self) -> bool:
        return False

    def transcribe(self, *, audio_path: str, file_name: str, max_tokens: Optional[int] = None,
                   tokenizer: Optional[str] = None, progress_cb: Optional[Callable] = None) -> AsrResult:
        raise RuntimeError(f"{NOT_CONFIGURED}: '{file_name}' was not transcribed")


def create_asr_provider(asr_cfg: "Optional[AsrConfig]") -> AsrProvider:
    """Factory: build a provider from config ``rag.asr``.

    - enabled: false → DisabledAsrProvider, whatever the rest says
    - no base_url (or no ``rag.asr`` section) → UnconfiguredAsrProvider: startup succeeds,
      audio uploads are refused with a message saying what to set
    - a provider removed in 1.5 → ValueError naming the migration guide (the config model
      refuses it first; this guards objects built around it)
    - unknown provider → ValueError
    """
    if asr_cfg is None:
        return UnconfiguredAsrProvider()
    if not getattr(asr_cfg, "enabled", True):
        return DisabledAsrProvider()

    provider = asr_cfg.provider
    if provider in REMOVED_ASR_PROVIDERS:
        raise ValueError(removed_asr_provider_message(provider))
    if provider != "openai-compatible":
        raise ValueError(f"Unknown rag.asr.provider: {provider!r} — the only provider is openai-compatible")
    if not asr_cfg.base_url:
        return UnconfiguredAsrProvider()
    return OpenAICompatibleAsrProvider(
        base_url=asr_cfg.base_url,
        api_key=asr_cfg.api_key,
        model=asr_cfg.model,
        request_timeout=float(getattr(asr_cfg, "timeout", 300.0)),
        max_retries=int(getattr(asr_cfg, "max_retries", 3)),
        segment_seconds=float(getattr(asr_cfg, "segment_seconds", 600.0)),
        language=getattr(asr_cfg, "language", None),
        chinese_conversion=getattr(asr_cfg, "chinese_conversion", "s2twp"),
        ca_file=getattr(asr_cfg, "ca_file", None),
    )


class SpeechToTextUnavailableError(ValueError):
    """An audio upload while speech-to-text is off or not configured (HTTP 422)."""


def current_asr_config():
    """The live ``rag.asr`` (console changes are applied to the config model in place)."""
    from src.config.config_manager import Config
    return getattr(getattr(Config.get_config_model(), "rag", None), "asr", None)


def check_audio_upload(file_name: str) -> None:
    """Raise SpeechToTextUnavailableError when ``file_name`` is audio that could not be transcribed."""
    reason = asr_upload_refusal(file_name, current_asr_config())
    if reason:
        raise SpeechToTextUnavailableError(reason)


def asr_upload_refusal(file_name: str, asr_cfg: "Optional[AsrConfig]") -> Optional[str]:
    """Why an upload of ``file_name`` must be refused now, or None.

    Only audio is checked: an audio file stored while speech-to-text is off or not configured
    could only fail later (or be indexed empty), so it is refused up front.
    """
    from src.domain.rag.audio_defense import _AUDIO_TRIM_EXTENSIONS
    if Path(file_name).suffix.lower() not in _AUDIO_TRIM_EXTENSIONS:
        return None
    if asr_cfg is not None and not getattr(asr_cfg, "enabled", True):
        return (f"'{file_name}' is audio, and speech-to-text is disabled (rag.asr.enabled=false). "
                "Enable rag.asr with an OpenAI-compatible endpoint to index audio.")
    if asr_cfg is None or not asr_cfg.base_url:
        return f"'{file_name}' is audio, and {NOT_CONFIGURED[0].lower()}{NOT_CONFIGURED[1:]}."
    return None
