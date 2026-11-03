
"""AsrProvider factory and implementations (docling local + openai-compatible cloud).

Covers: factory routing by config.provider (openai-compatible without base_url
errors immediately; unknown provider raises ValueError; fireredasr routes to
FireRedAsrProvider); DoclingWhisperAsrProvider wrapping docling_convert_once;
and OpenAICompatibleAsrProvider POSTing multipart to {base_url}/audio/transcriptions.
"""

from unittest.mock import patch

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
            timeout=30, max_retries=1, segment_seconds=120, language="zh",
        )
        p = create_asr_provider(cfg)
        assert isinstance(p, OpenAICompatibleAsrProvider)
        assert p._timeout == 30 and p._max_retries == 1
        assert p._segment_seconds == 120 and p._language == "zh"
        assert p.url == "http://stt:8000/v1/audio/transcriptions"

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
        # Disabled must not fall through to docling's whisper or
        # SimpleDirectoryReader (which still transcribed the audio).
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
        # The config-selected provider is the sole speech source: if unavailable
        # -> error explicitly so the file is marked failed (with an actionable
        # message), never silently fall back to no transcription (user decision)
        p = _FakeAsr(avail=False)
        loader = _loader(p)
        with patch(f"{_DL}.FileStorage.resolve_path", return_value=tmp_path / "m.mp3"):
            (tmp_path / "m.mp3").write_bytes(b"x")
            with pytest.raises(RuntimeError, match="not available"):
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
