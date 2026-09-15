
"""FireRedAsrProvider unit tests (no weights/GPU).

Engineering constraints under test:
- 60s hard cap: every planned segment is ≤ MAX_SEGMENT_SECONDS.
- Cut points are the midpoint of a silence; no silence in the window → hard cut
  (never emit a >60s segment).
- Output passes through OpenCC s2twp (Simplified → Traditional, Taiwan wording).
- Weights missing → available() is False (document_loader falls back, no crash).

Model inference (_transcribe_segments) is mocked — real-model acceptance runs
in the evals.
"""

import wave
from unittest.mock import patch

from src.domain.rag.fireredasr_provider import (
    MAX_SEGMENT_SECONDS,
    FireRedAsrProvider,
    parse_silences,
    plan_segments,
)

_FP = "src.domain.rag.fireredasr_provider"


class TestParseSilences:
    def test_extracts_pairs(self):
        stderr = (
            "[silencedetect @ 0x1] silence_start: 12.5\n"
            "[silencedetect @ 0x1] silence_end: 13.1 | silence_duration: 0.6\n"
            "[silencedetect @ 0x1] silence_start: 40.0\n"
            "[silencedetect @ 0x1] silence_end: 40.8 | silence_duration: 0.8\n"
        )
        assert parse_silences(stderr) == [(12.5, 13.1), (40.0, 40.8)]

    def test_trailing_unpaired_start_dropped(self):
        # File ends mid-silence: a trailing start with no end → dropped by zip, no crash
        stderr = "silence_start: 5.0\nsilence_end: 5.5\nsilence_start: 58.0\n"
        assert parse_silences(stderr) == [(5.0, 5.5)]

    def test_empty(self):
        assert parse_silences("no silence lines here") == []


class TestPlanSegments:
    def test_short_audio_single_segment(self):
        assert plan_segments(30.0, [(10.0, 10.5)]) == [(0.0, 30.0)]

    def test_cuts_at_silence_midpoint(self):
        # 120s, silences at 50-51s and 100-101s → cut at 50.5 / 100.5
        segs = plan_segments(120.0, [(50.0, 51.0), (100.0, 101.0)])
        assert segs == [(0.0, 50.5), (50.5, 100.5), (100.5, 120.0)]

    def test_no_silence_hard_cut(self):
        # No silence throughout → hard cut at 55s (rather cut a word than feed >60s)
        segs = plan_segments(120.0, [])
        assert segs == [(0.0, 55.0), (55.0, 110.0), (110.0, 120.0)]

    def test_every_segment_within_hard_limit(self):
        # Adversarially distributed silences (all bunched at the start) must not produce an over-limit segment
        segs = plan_segments(300.0, [(1.0, 1.4), (2.0, 2.4)])
        assert all(e - s <= MAX_SEGMENT_SECONDS + 1e-9 for s, e in segs)
        # Segments are seamless and cover the full length
        assert segs[0][0] == 0.0 and segs[-1][1] == 300.0
        assert all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))

    def test_silence_beyond_window_ignored(self):
        # The only silence is at 70s (beyond the 55s window) → no candidate in the window, hard cut at 55
        segs = plan_segments(80.0, [(70.0, 70.5)])
        assert segs[0] == (0.0, 55.0)


class TestFireRedAsrProvider:
    def test_available_false_without_weights(self):
        with patch(f"{_FP}._resolve_model_dir", return_value=None):
            assert FireRedAsrProvider().available() is False

    def test_transcribe_joins_segments_and_converts_to_traditional(self, tmp_path):
        # A real wav (2s silence, 16k mono) goes through real ffmpeg preprocessing; only model inference is mocked.
        # Segment text is Simplified — assert the output has been converted to Traditional via s2twp (with Taiwan wording).
        src = tmp_path / "in.wav"
        with wave.open(str(src), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000 * 2)

        with patch.object(
            FireRedAsrProvider, "_transcribe_segments",
            return_value=["这是软件测试", "内存很大"],
        ):
            r = FireRedAsrProvider().transcribe(
                audio_path=str(src), file_name="in.wav")
        assert r.text == "這是軟體測試\n記憶體很大"
        assert r.chunks is None  # AED has no punctuation or structure → leaf_splitter plain-text path

    def test_transcribe_reports_progress(self, tmp_path):
        src = tmp_path / "in.wav"
        with wave.open(str(src), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)

        seen = []
        with patch.object(
            FireRedAsrProvider, "_transcribe_segments", return_value=["文字"],
        ):
            FireRedAsrProvider().transcribe(
                audio_path=str(src), file_name="in.wav",
                progress_cb=lambda stage, done, total: seen.append((stage, done, total)),
            )
        assert seen == [("loading", 1, 1)]


class TestModelPlumbing:
    def test_resolve_model_dir_requires_weight_file(self, tmp_path):
        import src.domain.rag.fireredasr_provider as fp
        with patch(f"{_FP}.resolve_assets_dir", create=True), \
             patch("src.domain.rag.docling_loader.resolve_assets_dir",
                   return_value=tmp_path):
            assert fp._resolve_model_dir() is None  # directory does not exist
            d = tmp_path / "fireredasr" / "FireRedASR-AED-L"
            d.mkdir(parents=True)
            assert fp._resolve_model_dir() is None  # directory exists but model.pth.tar is missing
            (d / "model.pth.tar").write_bytes(b"x")
            assert fp._resolve_model_dir() == d
        with patch("src.domain.rag.docling_loader.resolve_assets_dir",
                   return_value=None):
            assert fp._resolve_model_dir() is None  # assets missing entirely

    def test_load_model_singleton(self, tmp_path, monkeypatch):
        # fireredasr is a uv path dependency (vendor/fireredasr_src) — a normal import
        import src.domain.rag.fireredasr_provider as fp
        monkeypatch.setattr(fp, "_model", None)  # clear the singleton (restored automatically after the test)
        fake = object()
        with patch("fireredasr.models.fireredasr.FireRedAsr") as M, \
             patch("torch.cuda.is_available", return_value=False):
            M.from_pretrained.return_value = fake
            m1 = fp._load_model(tmp_path)
            m2 = fp._load_model(tmp_path)
        assert m1 == (fake, False)
        assert m2 is m1  # singleton: the second call does not reload
        M.from_pretrained.assert_called_once_with("aed", str(tmp_path))

    def test_transcribe_segments_per_segment_calls(self, tmp_path, monkeypatch):
        import src.domain.rag.fireredasr_provider as fp
        fake_model = type("M", (), {})()
        calls = []

        def fake_transcribe(uttids, paths, args):
            calls.append((uttids, paths, args))
            return [{"text": f" 第{len(calls)}段 "}]

        fake_model.transcribe = fake_transcribe
        with patch(f"{_FP}._resolve_model_dir", return_value=tmp_path), \
             patch(f"{_FP}._load_model", return_value=(fake_model, False)):
            out = FireRedAsrProvider()._transcribe_segments(
                [tmp_path / "seg_0000.wav", tmp_path / "seg_0001.wav"])
        assert out == ["第1段", "第2段"]  # stripped
        assert len(calls) == 2  # per-segment (not batched)
        assert calls[0][2]["use_gpu"] is False

    def test_transcribe_segments_raises_without_weights(self):
        with patch(f"{_FP}._resolve_model_dir", return_value=None):
            import pytest
            with pytest.raises(RuntimeError, match="權重"):
                FireRedAsrProvider()._transcribe_segments([])

    def test_resample_failure_raises(self, tmp_path):
        from src.domain.rag.fireredasr_provider import _resample_to_wav16k
        import pytest
        bogus = tmp_path / "not_audio.bin"
        bogus.write_bytes(b"not an audio file")
        with pytest.raises(RuntimeError, match="ffmpeg"):
            _resample_to_wav16k(str(bogus), tmp_path / "out.wav")
