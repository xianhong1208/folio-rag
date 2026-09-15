
"""ref -- AsrProvider factory and implementations (docling local + openai-compatible cloud).

Covers: factory routing by config.provider (openai-compatible without base_url
errors immediately; unknown provider raises ValueError; fireredasr routes to
FireRedAsrProvider); DoclingWhisperAsrProvider wrapping docling_convert_once;
and OpenAICompatibleAsrProvider POSTing multipart to {base_url}/audio/transcriptions.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config.model import AsrConfig
from src.domain.rag.asr_provider import (
    AsrResult,
    DoclingWhisperAsrProvider,
    OpenAICompatibleAsrProvider,
    create_asr_provider,
)

_AP = "src.domain.rag.asr_provider"


class TestFactory:
    def test_default_docling_whisper(self):
        p = create_asr_provider(AsrConfig())
        assert isinstance(p, DoclingWhisperAsrProvider)

    def test_none_config_gives_default(self):
        """rag.asr unset (None) -> default provider (backward compat)."""
        assert isinstance(create_asr_provider(None), DoclingWhisperAsrProvider)

    def test_openai_compatible_requires_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            create_asr_provider(AsrConfig(provider="openai-compatible"))

    def test_openai_compatible_built_with_config(self):
        cfg = AsrConfig(
            provider="openai-compatible",
            base_url="http://stt:8000/v1", api_key="k", model="whisper-1",
        )
        p = create_asr_provider(cfg)
        assert isinstance(p, OpenAICompatibleAsrProvider)

    def test_fireredasr_routes_to_provider(self):
        from src.domain.rag.fireredasr_provider import FireRedAsrProvider
        p = create_asr_provider(AsrConfig(provider="fireredasr"))
        assert isinstance(p, FireRedAsrProvider)

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="bogus"):
            create_asr_provider(AsrConfig(provider="bogus"))


class TestDoclingProvider:
    def test_delegates_to_docling_convert_once(self):
        cb = object()
        # convert_once now returns chunk-record dicts; audio has no heading/page,
        # so the provider must flatten to plain text (also compatible with the old bare-string form)
        records = [
            {"text": "c1", "headings": None, "page_no": None},
            "c2",
        ]
        with patch(f"{_AP}.docling_convert_once",
                   return_value=("full text", records)) as m:
            r = DoclingWhisperAsrProvider().transcribe(
                audio_path="/tmp/a.mp3", file_name="a.mp3",
                max_tokens=380, tokenizer="intfloat/multilingual-e5-large",
                progress_cb=cb,
            )
        assert isinstance(r, AsrResult)
        assert r.text == "full text"
        assert r.chunks == ["c1", "c2"]
        m.assert_called_once_with(
            file_path="/tmp/a.mp3", file_name="a.mp3",
            max_tokens=380, tokenizer="intfloat/multilingual-e5-large",
            progress_cb=cb,
        )

    def test_default_max_tokens_not_none(self):
        # Called without max_tokens (the shape of the evals draft script) -- None
        # must not be passed straight into convert_once (it would override the
        # default 512 -> None//2 TypeError inside refine)
        with patch(f"{_AP}.docling_convert_once",
                   return_value=("t", [])) as m:
            DoclingWhisperAsrProvider().transcribe(
                audio_path="/tmp/a.mp3", file_name="a.mp3")
        assert m.call_args.kwargs["max_tokens"] == 512


def _mock_httpx_post(json_body=None, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_body or {"text": "雲端轉錄結果"}
    if status >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=resp)
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = resp
    return client


class TestOpenAICompatibleProvider:
    def _provider(self):
        return OpenAICompatibleAsrProvider(
            base_url="http://stt:8000/v1", api_key="sk-test",
            model="whisper-1", request_timeout=120.0,
        )

    def test_posts_multipart_to_transcriptions(self, tmp_path):
        wav = tmp_path / "m.wav"
        wav.write_bytes(b"RIFFfake")
        client = _mock_httpx_post()
        with patch(f"{_AP}.httpx.Client", return_value=client):
            r = self._provider().transcribe(audio_path=str(wav), file_name="m.wav")

        assert r.text == "雲端轉錄結果"
        assert r.chunks is None  # cloud has no pre-split chunks, handed back to leaf_splitter
        url = client.post.call_args.args[0]
        kw = client.post.call_args.kwargs
        assert url == "http://stt:8000/v1/audio/transcriptions"
        assert kw["data"]["model"] == "whisper-1"
        assert "file" in kw["files"]
        assert kw["files"]["file"][0] == "m.wav"

    def test_bearer_auth_header(self, tmp_path):
        wav = tmp_path / "m.wav"; wav.write_bytes(b"x")
        client = _mock_httpx_post()
        with patch(f"{_AP}.httpx.Client", return_value=client) as mk:
            self._provider().transcribe(audio_path=str(wav), file_name="m.wav")
        headers = mk.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-test"

    def test_no_api_key_no_auth_header(self, tmp_path):
        """Self-hosted services often have no key: must not send an empty Bearer."""
        wav = tmp_path / "m.wav"; wav.write_bytes(b"x")
        client = _mock_httpx_post()
        p = OpenAICompatibleAsrProvider(base_url="http://stt:8000/v1")
        with patch(f"{_AP}.httpx.Client", return_value=client) as mk:
            p.transcribe(audio_path=str(wav), file_name="m.wav")
        assert "Authorization" not in (mk.call_args.kwargs.get("headers") or {})

    def test_http_error_raises(self, tmp_path):
        """Cloud 4xx/5xx raise as-is -- the upstream index_document takes the existing failure path and writes a failed tag."""
        import httpx
        wav = tmp_path / "m.wav"; wav.write_bytes(b"x")
        client = _mock_httpx_post(status=500)
        with patch(f"{_AP}.httpx.Client", return_value=client):
            with pytest.raises(httpx.HTTPStatusError):
                self._provider().transcribe(audio_path=str(wav), file_name="m.wav")

    def test_base_url_trailing_slash_normalized(self, tmp_path):
        wav = tmp_path / "m.wav"; wav.write_bytes(b"x")
        client = _mock_httpx_post()
        p = OpenAICompatibleAsrProvider(base_url="http://stt:8000/v1/")
        with patch(f"{_AP}.httpx.Client", return_value=client):
            p.transcribe(audio_path=str(wav), file_name="m.wav")
        assert client.post.call_args.args[0] == "http://stt:8000/v1/audio/transcriptions"


_DL = "src.domain.rag.document_loader"


class _FakeAsr:
    def __init__(self, text="轉錄文字", chunks=None, avail=True, err=None):
        self._r = AsrResult(text=text, chunks=chunks)
        self._avail = avail
        self._err = err
        self.calls = []

    def available(self):
        return self._avail

    def transcribe(self, **kw):
        self.calls.append(kw)
        if self._err:
            raise self._err
        return self._r


def _loader(provider, enabled=True):
    from src.domain.rag.document_loader import DocumentLoader
    return DocumentLoader(
        leaf_chunk_size=256, embedding_model_name="e5",
        asr_provider=provider, asr_enabled=enabled,
    )


class TestDocumentLoaderAudioRouting:
    def _load(self, loader, tmp_path):
        wav = tmp_path / "m.mp3"
        wav.write_bytes(b"x")
        with patch(f"{_DL}.FileStorage.resolve_path", return_value=wav), \
             patch(f"{_DL}._trim_audio_leading_silence", return_value=None):
            return loader.load(file_path="storage/m.mp3", file_id="f1", file_name="m.mp3")

    def test_audio_routes_to_provider_with_chunks(self, tmp_path):
        p = _FakeAsr(text="會議內容", chunks=["c1", "c2"])
        doc = self._load(_loader(p), tmp_path)
        assert doc.text == "會議內容"
        assert doc.metadata["_docling_chunks"] == ["c1", "c2"]
        assert p.calls and p.calls[0]["file_name"] == "m.mp3"
        assert p.calls[0]["max_tokens"] == 256  # min(leaf, 450), no context-gen

    def test_cloud_provider_no_chunks_no_docling_chunks_key(self, tmp_path):
        doc = self._load(_loader(_FakeAsr(chunks=None)), tmp_path)
        assert "_docling_chunks" not in doc.metadata  # takes the leaf_splitter plain-text path

    def test_hallucination_filter_applied_after_provider(self, tmp_path):
        p = _FakeAsr(text="開始。打赏支持本栏目。結束")
        doc = self._load(_loader(p), tmp_path)
        assert "打赏" not in doc.text and "開始" in doc.text

    def test_disabled_asr_skips_provider(self, tmp_path):
        p = _FakeAsr()
        loader = _loader(p, enabled=False)
        with patch(f"{_DL}.is_docling_supported", return_value=False), \
             patch.object(type(loader), "_load_via_simple_reader",
                          return_value="FALLBACK") as fb, \
             patch(f"{_DL}.FileStorage.resolve_path", return_value=tmp_path / "m.mp3"):
            (tmp_path / "m.mp3").write_bytes(b"x")
            out = loader.load(file_path="s/m.mp3", file_id="f1", file_name="m.mp3")
        assert out == "FALLBACK" and p.calls == []
        fb.assert_called_once()

    def test_unavailable_provider_raises_clear_error(self, tmp_path):
        # The config-selected provider is the sole speech source: if unavailable
        # -> error explicitly so the file is marked failed (with an actionable
        # message), never silently fall back to no transcription (user decision)
        p = _FakeAsr(avail=False)
        loader = _loader(p)
        with patch(f"{_DL}.FileStorage.resolve_path", return_value=tmp_path / "m.mp3"):
            (tmp_path / "m.mp3").write_bytes(b"x")
            with pytest.raises(RuntimeError, match="不可用"):
                loader.load(file_path="s/m.mp3", file_id="f1", file_name="m.mp3")
        assert p.calls == []  # transcription was not attempted

    def test_provider_error_propagates(self, tmp_path):
        p = _FakeAsr(err=RuntimeError("stt down"))
        with pytest.raises(RuntimeError, match="stt down"):
            self._load(_loader(p), tmp_path)
