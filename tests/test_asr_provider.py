
"""AsrProvider factory, the removed in-process providers, the upload refusal and the
Chinese-script conversion (1.5: speech-to-text only through an OpenAI-compatible endpoint).

Covers: factory routing (no base_url -> unconfigured, never a crash at startup;
disabled; removed providers refused with the migration guide named); the config
model refusing docling-whisper / fireredasr; asr_upload_refusal for audio only;
convert_chinese (s2twp for Chinese, English untouched); the ca_file trust.
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.model import AsrConfig
from src.domain.rag.asr_provider import (
    AsrResult,
    DisabledAsrProvider,
    OpenAICompatibleAsrProvider,
    SpeechToTextUnavailableError,
    UnconfiguredAsrProvider,
    asr_upload_refusal,
    check_audio_upload,
    create_asr_provider,
    tls_verify,
)

_AP = "src.domain.rag.asr_provider"


class TestFactory:
    def test_default_is_openai_compatible_and_unconfigured(self):
        """No base_url: the service still starts; audio is refused instead (no crash at startup)."""
        assert AsrConfig().provider == "openai-compatible"
        p = create_asr_provider(AsrConfig())
        assert isinstance(p, UnconfiguredAsrProvider) and p.available() is False
        with pytest.raises(RuntimeError, match="not configured.*ASR_BASE_URL.*migrate-asr"):
            p.transcribe(audio_path="a.wav", file_name="a.wav")

    def test_none_config_is_unconfigured(self):
        assert isinstance(create_asr_provider(None), UnconfiguredAsrProvider)

    def test_openai_compatible_built_with_config(self):
        cfg = AsrConfig(
            base_url="http://stt:8000/v1", api_key="k", model="whisper-1",
            timeout=30, max_retries=1, segment_seconds=120, language="zh",
            chinese_conversion="none", ca_file="/ca.pem",
        )
        p = create_asr_provider(cfg)
        assert isinstance(p, OpenAICompatibleAsrProvider) and p.available()
        assert p._timeout == 30 and p._max_retries == 1
        assert p._segment_seconds == 120 and p._language == "zh"
        assert p._chinese_conversion == "none" and p._ca_file == "/ca.pem"
        assert p.url == "http://stt:8000/v1/audio/transcriptions"

    def test_disabled_needs_no_endpoint(self):
        """enabled: false (the cloud profile's default section, without base_url) builds a provider
        that transcribes nothing, instead of failing every indexing job and query."""
        p = create_asr_provider(AsrConfig(enabled=False))
        assert isinstance(p, DisabledAsrProvider) and p.available() is False
        with pytest.raises(RuntimeError, match="disabled"):
            p.transcribe(audio_path="a.wav", file_name="a.wav")

    def test_removed_provider_on_a_built_object_is_refused(self):
        cfg = AsrConfig(base_url="http://x/v1").model_copy(update={"provider": "fireredasr"})
        with pytest.raises(ValueError, match="removed in Folio 1.5.*migrate-asr"):
            create_asr_provider(cfg)
        cfg = AsrConfig(base_url="http://x/v1").model_copy(update={"provider": "bogus"})
        with pytest.raises(ValueError, match="bogus"):
            create_asr_provider(cfg)


class TestRemovedProviders:
    @pytest.mark.parametrize("name", ["fireredasr", "docling-whisper", " FireRedASR "])
    def test_config_refuses_removed_providers_with_the_guide(self, name):
        with pytest.raises(ValidationError, match="removed in Folio 1.5.*docs/migrate-asr.md"):
            AsrConfig(provider=name)

    def test_unknown_provider_refused(self):
        with pytest.raises(ValidationError, match="only provider is openai-compatible"):
            AsrConfig(provider="bogus")

    @pytest.mark.parametrize("value", [None, "", "OpenAI-Compatible"])
    def test_blank_or_cased_means_openai_compatible(self, value):
        assert AsrConfig(provider=value).provider == "openai-compatible"

    def test_chinese_conversion_values(self):
        assert AsrConfig().chinese_conversion == "s2twp"
        assert AsrConfig(chinese_conversion=" NONE ").chinese_conversion == "none"
        assert AsrConfig(chinese_conversion=None).chinese_conversion == "none"
        with pytest.raises(ValidationError, match="s2twp or none"):
            AsrConfig(chinese_conversion="s2t")


class TestUploadRefusal:
    @pytest.mark.parametrize("name", ["a.mp3", "b.WAV", "c.m4a", "d.flac", "e.ogg", "f.aac"])
    def test_audio_refused_without_endpoint(self, name):
        reason = asr_upload_refusal(name, AsrConfig())
        assert reason and name in reason and "rag.asr.base_url" in reason

    def test_audio_refused_when_disabled(self):
        reason = asr_upload_refusal("a.mp3", AsrConfig(enabled=False, base_url="http://x/v1"))
        assert reason and "disabled" in reason

    def test_audio_refused_without_section(self):
        assert asr_upload_refusal("a.mp3", None)

    def test_configured_audio_and_other_files_pass(self):
        assert asr_upload_refusal("a.mp3", AsrConfig(base_url="http://x/v1")) is None
        assert asr_upload_refusal("a.pdf", AsrConfig()) is None
        assert asr_upload_refusal("notes.txt", None) is None

    def test_check_reads_the_live_config(self):
        with patch(f"{_AP}.current_asr_config", return_value=AsrConfig()):
            with pytest.raises(SpeechToTextUnavailableError, match="not configured"):
                check_audio_upload("a.mp3")
            check_audio_upload("a.pdf")
        with patch(f"{_AP}.current_asr_config", return_value=AsrConfig(base_url="http://x/v1")):
            check_audio_upload("a.mp3")

    def test_refusal_is_a_value_error(self):
        """The REST layer maps it to 422; batch uploads list it as a failed file."""
        assert issubclass(SpeechToTextUnavailableError, ValueError)

    def test_current_config_follows_the_config_model(self):
        from src.domain.rag.asr_provider import current_asr_config
        cfg = AsrConfig(base_url="http://x/v1")

        class _Rag:
            asr = cfg

        class _Model:
            rag = _Rag()
        with patch("src.config.config_manager.Config.get_config_model", return_value=_Model()):
            assert current_asr_config() is cfg


class TestChineseConversion:
    def test_s2twp_converts_simplified(self):
        from src.domain.rag.han_script import convert_chinese
        assert convert_chinese("今天天气很好，我们去看视频。", "s2twp") == "今天天氣很好，我們去看影片。"

    def test_english_and_none_pass_through(self):
        from src.domain.rag.han_script import convert_chinese
        text = "The quick brown fox jumps over the lazy dog, 42 times."
        assert convert_chinese(text, "s2twp") == text
        assert convert_chinese("今天天气", "none") == "今天天气"
        assert convert_chinese("", "s2twp") == ""

    def test_unknown_mode_is_an_error(self):
        from src.domain.rag.han_script import convert_chinese
        with pytest.raises(ValueError):
            convert_chinese("天气", "s2t")


class TestTlsVerify:
    def test_no_ca_file_keeps_default_trust(self):
        assert tls_verify(None) is True and tls_verify("") is True

    def test_ca_file_is_added_to_default_trust(self, tmp_path):
        import ssl
        import subprocess
        key, crt = tmp_path / "k.pem", tmp_path / "c.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
                        "-out", str(crt), "-days", "1", "-subj", "/CN=test-ca"],
                       check=True, capture_output=True)
        import httpx
        default = len(httpx.create_ssl_context().get_ca_certs())
        ctx = tls_verify(str(crt))
        assert isinstance(ctx, ssl.SSLContext)
        subjects = [c["subject"] for c in ctx.get_ca_certs()]
        assert ((("commonName", "test-ca"),),) in subjects
        assert len(subjects) >= default  # added, not replaced

    def test_missing_ca_file_is_an_error_not_unverified(self, tmp_path):
        with pytest.raises(RuntimeError, match="ca_file"):
            tls_verify(str(tmp_path / "missing.pem"))


# OpenAICompatibleAsrProvider (segmented upload, retries, content guard) is covered in
# tests/test_cloud_asr.py against the in-process fake server.


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

    def test_disabled_asr_fails_the_file_without_transcribing(self, tmp_path):
        # Disabled must not fall through to docling or SimpleDirectoryReader.
        p = _FakeAsr()
        loader = _loader(p, enabled=False)
        with patch(f"{_DL}.is_docling_supported", return_value=True), \
             patch(f"{_DL}.docling_convert_once") as conv, \
             patch.object(type(loader), "_load_via_simple_reader") as fb, \
             patch(f"{_DL}.FileStorage.resolve_path", return_value=tmp_path / "m.mp3"):
            (tmp_path / "m.mp3").write_bytes(b"x")
            with pytest.raises(ValueError, match="Speech-to-text is disabled"):
                loader.load(file_path="s/m.mp3", file_id="f1", file_name="m.mp3")
        assert p.calls == []
        conv.assert_not_called()
        fb.assert_not_called()

    def test_unavailable_provider_raises_clear_error(self, tmp_path):
        # An audio file stored before speech-to-text lost its endpoint: error explicitly
        # so the file is marked failed (with what to set), never an empty index entry.
        p = _FakeAsr(avail=False)
        loader = _loader(p)
        with patch(f"{_DL}.FileStorage.resolve_path", return_value=tmp_path / "m.mp3"):
            (tmp_path / "m.mp3").write_bytes(b"x")
            with pytest.raises(RuntimeError, match="not configured.*ASR_BASE_URL"):
                loader.load(file_path="s/m.mp3", file_id="f1", file_name="m.mp3")
        assert p.calls == []  # transcription was not attempted

    def test_provider_error_propagates(self, tmp_path):
        p = _FakeAsr(err=RuntimeError("stt down"))
        with pytest.raises(RuntimeError, match="stt down"):
            self._load(_loader(p), tmp_path)


class TestTimeBlocks:
    def test_segments_group_into_five_minute_blocks(self):
        from src.domain.rag.audio_segments import time_block_records
        segs = [(i * 55.0, (i + 1) * 55.0, f"seg{i}") for i in range(12)]  # 0 .. 660 s
        recs = time_block_records(segs)
        heads = [r["headings"][0] for r in recs]
        assert heads[:6] == ["00:00:00–00:05:30"] * 6       # starts 0..275 s
        assert heads[6:11] == ["00:05:30–00:10:05"] * 5     # starts 330..550 s
        assert heads[11] == "00:10:05–00:11:00"
        assert [r["text"] for r in recs] == [f"seg{i}" for i in range(12)]

    def test_untimed_and_empty(self):
        from src.domain.rag.audio_segments import fmt_timestamp, time_block_records
        assert time_block_records([(0.0, 0.0, "whole file")]) is None
        assert time_block_records([]) is None
        assert time_block_records([(0.0, 5.0, "  "), (5.0, 9.0, "x")]) == [
            {"text": "x", "headings": ["00:00:05–00:00:09"]}]
        assert fmt_timestamp(3725.9) == "01:02:05"

    def test_loader_sizes_timed_records_within_blocks(self, tmp_path):
        """Timed records become leaf-sized chunks that never cross a time block, and
        leaves group into one parent per block."""
        from src.domain.rag.audio_segments import time_block_records
        from src.domain.rag.hierarchy import build_hierarchy
        from src.domain.rag.leaf_splitter import LeafSplitter

        line = "我們今天討論第三季的營運計畫以及預算分配的細節後續再由各部門回報進度"
        segs = [(i * 55.0, (i + 1) * 55.0, line * 4) for i in range(12)]
        p = _FakeAsr(text="\n".join(t for *_, t in segs), chunks=time_block_records(segs))
        doc = self._load(_loader(p), tmp_path)
        chunks = doc.metadata["_docling_chunks"]
        assert all(isinstance(c, dict) and c["headings"] for c in chunks)
        assert len(chunks) < 12  # short segments merged within their block
        leaves = LeafSplitter(leaf_chunk_size=256, chunk_overlap=0).split_into_leaf_documents(doc)
        _, parents = build_hierarchy(leaves, parent_target_tokens=2048)
        assert [p.metadata["headings"] for p in parents] == [
            ["00:00:00–00:05:30"], ["00:05:30–00:10:05"], ["00:10:05–00:11:00"]]

    def _load(self, loader, tmp_path):
        f = tmp_path / "m.mp3"
        f.write_bytes(b"x")
        with patch(f"{_DL}.FileStorage.resolve_path", return_value=f), \
                patch(f"{_DL}._trim_audio_leading_silence", return_value=None):
            return loader.load(file_path=str(f), file_id="1", file_name="m.mp3")


def test_oversized_timed_segment_is_split_to_budget(tmp_path):
    from src.domain.rag.audio_segments import time_block_records
    from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer

    tok = get_chunk_tokenizer(None)
    line = "我們今天討論第三季的營運計畫以及預算分配的細節後續再由各部門回報進度"
    p = _FakeAsr(text=line * 30, chunks=time_block_records([(0.0, 55.0, line * 30)]))
    doc = TestTimeBlocks()._load(_loader(p), tmp_path)
    chunks = doc.metadata["_docling_chunks"]
    assert len(chunks) > 1
    assert all(len(tok.encode(c["text"])) <= 256 for c in chunks)
    assert "".join(c["text"] for c in chunks) == line * 30
