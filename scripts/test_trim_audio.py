#!/usr/bin/env python3
"""Compare leading-silence trimming against Whisper language detection/transcription.

Trims a recording's leading silence with ffmpeg, then reports duration cut,
detected language, and a short transcript before vs. after trimming, so
--threshold / --duration can be tuned before wiring trimming into indexing.

Usage:
    uv run python scripts/test_trim_audio.py path/to/meeting.mp3 \\
        --threshold -35 --duration 2.0 --keep-trimmed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, errors="replace", check=True,
    )
    return float(result.stdout.strip())


def _codec_args_for(ext: str) -> list[str]:
    if ext == ".mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2"]
    if ext == ".wav":
        return ["-c:a", "pcm_s16le"]
    return []  # let ffmpeg pick


def detect_leading_silence_end(
    src: Path, *, threshold_db: float, min_duration: float
) -> float:
    """Use ffmpeg silencedetect to find where leading silence ends (= audio starts).

    Returns 0.0 if the file does not start with silence (first silent block > 1s into file),
    or no silence is detected at all.

    Why silencedetect instead of silenceremove: silenceremove in ffmpeg 4.x has known
    reliability issues on long files (>1h) — sometimes cuts everything. silencedetect
    just reports timestamps without mutating audio, then we seek with -ss (rock solid).
    """
    af = f"silencedetect=noise={threshold_db}dB:duration={min_duration}"
    cmd = ["ffmpeg", "-hide_banner", "-i", str(src), "-vn", "-af", af, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    # silencedetect emits to stderr regardless of -loglevel; parse there
    starts = re.findall(r"silence_start:\s*([\d.]+)", result.stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", result.stderr)
    if not starts or not ends:
        return 0.0
    if float(starts[0]) > 1.0:
        # file starts with audio, not silence — nothing to trim
        return 0.0
    return float(ends[0])


def trim_leading_silence(
    src: Path, dst: Path, *, threshold_db: float, duration_s: float, detection: str
) -> float:
    """Trim leading silence and write to dst. Returns the seek timestamp used.

    detection arg is accepted for API compatibility but currently silencedetect uses
    its default (rms) since that worked reliably for the meeting-recording cases tested.
    """
    seek_ts = detect_leading_silence_end(
        src, threshold_db=threshold_db, min_duration=duration_s
    )

    # -ss before -i = fast seek (keyframe-aligned, very quick); -vn drops embedded artwork
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(seek_ts),
        "-i", str(src), "-vn",
        *_codec_args_for(dst.suffix.lower()), str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg seek-trim failed (exit {result.returncode}):\n{result.stderr}")

    size = dst.stat().st_size if dst.exists() else 0
    if size < 4096:
        raise RuntimeError(
            f"Output near-empty ({size} bytes) after seeking to {seek_ts:.2f}s. "
            f"Original file is only {seek_ts:.2f}s long? Or seek failed.\n"
            f"ffmpeg stderr: {result.stderr or '(empty)'}"
        )

    return seek_ts


def truncate_audio(src: Path, dst: Path, seconds: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
        "-t", str(seconds), *_codec_args_for(dst.suffix.lower()), str(dst),
    ]
    subprocess.run(cmd, check=True)


def load_whisper(device: str):
    try:
        import whisper
    except ImportError:
        sys.exit("[error] openai-whisper not installed; run `uv sync` first")

    whisper_dir = PROJECT_ROOT / "assets" / "whisper_models"
    if whisper_dir.is_dir() and any(whisper_dir.glob("*.pt")):
        print(f"[whisper] loading turbo from {whisper_dir} (device={device})")
        return whisper.load_model("turbo", device=device, download_root=str(whisper_dir))
    print(f"[whisper] no local model in {whisper_dir} -- will fetch from HF Hub")
    return whisper.load_model("turbo", device=device)


def analyze(model, audio_path: Path, preview_seconds: float, device: str):
    """Detect language on first 30s + transcribe first preview_seconds for human inspection."""
    import whisper

    audio = whisper.load_audio(str(audio_path))
    clip30 = whisper.pad_or_trim(audio)
    # turbo / large-v3 use 128 mel bins; older Whisper uses 80. Pull from model dims.
    n_mels = getattr(model.dims, "n_mels", 80)
    mel = whisper.log_mel_spectrogram(clip30, n_mels=n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

    tmp = tempfile.NamedTemporaryFile(suffix=audio_path.suffix, delete=False)
    tmp.close()
    preview_path = Path(tmp.name)
    try:
        truncate_audio(audio_path, preview_path, preview_seconds)
        # fp16 only valid on CUDA; force off on CPU to avoid warnings/errors
        result = model.transcribe(
            str(preview_path),
            language=detected,
            verbose=False,
            fp16=(device != "cpu"),
        )
        text = result.get("text", "").strip()
    finally:
        preview_path.unlink(missing_ok=True)

    return detected, top5, text


def fmt_seconds(s: float) -> str:
    m, sec = divmod(s, 60)
    return f"{int(m)}m{sec:05.2f}s"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Audio file (.mp3 / .wav)")
    parser.add_argument("--threshold", type=float, default=-35.0,
                        help="silenceremove start_threshold in dB (default: -35)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="silenceremove start_duration in seconds (default: 2.0)")
    parser.add_argument("--detection", default="peak", choices=["peak", "rms"],
                        help="silenceremove detection mode (default: peak)")
    parser.add_argument("--preview-seconds", type=float, default=60.0,
                        help="seconds to transcribe for the preview (default: 60)")
    parser.add_argument("--device", default="cpu",
                        help="Whisper device — keep cpu on gfx1151 (default: cpu)")
    parser.add_argument("--keep-trimmed", action="store_true",
                        help="Save trimmed file next to input for inspection")
    args = parser.parse_args()

    src = args.input.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"[error] file not found: {src}")

    print(f"[input] {src}")
    orig_dur = ffprobe_duration(src)
    print(f"[input] duration={fmt_seconds(orig_dur)}")

    if args.keep_trimmed:
        trimmed = src.with_name(f"{src.stem}_trimmed{src.suffix}")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=src.suffix, delete=False)
        tmp.close()
        trimmed = Path(tmp.name)

    try:
        print(f"[trim] threshold={args.threshold}dB  duration={args.duration}s  (silencedetect + seek)")
        seek_ts = trim_leading_silence(src, trimmed, threshold_db=args.threshold,
                                       duration_s=args.duration, detection=args.detection)
        if seek_ts == 0.0:
            print(f"[trim] no leading silence detected (file starts with audio) -- no trim applied")
        else:
            print(f"[trim] leading silence ends at {fmt_seconds(seek_ts)} -- seeking past it")

        new_dur = ffprobe_duration(trimmed)
        cut = orig_dur - new_dur
        cut_pct = (cut / orig_dur * 100) if orig_dur > 0 else 0
        print(f"[trim] {fmt_seconds(orig_dur)} -> {fmt_seconds(new_dur)}  "
              f"(cut {fmt_seconds(cut)} = {cut_pct:.1f}%)")

        model = load_whisper(args.device)

        print("\n----- BEFORE trim -----")
        a_lang, a_top5, a_text = analyze(model, src, args.preview_seconds, args.device)
        print(f"  detected language: {a_lang}")
        print(f"  top5: {', '.join(f'{l}={p:.1%}' for l, p in a_top5)}")
        print(f"  first {args.preview_seconds:.0f}s transcript:")
        print(f"  > {a_text[:300]}{'...' if len(a_text) > 300 else ''}")

        print("\n----- AFTER trim -----")
        b_lang, b_top5, b_text = analyze(model, trimmed, args.preview_seconds, args.device)
        print(f"  detected language: {b_lang}")
        print(f"  top5: {', '.join(f'{l}={p:.1%}' for l, p in b_top5)}")
        print(f"  first {args.preview_seconds:.0f}s transcript:")
        print(f"  > {b_text[:300]}{'...' if len(b_text) > 300 else ''}")

        print("\n----- VERDICT -----")
        if a_lang != b_lang:
            print(f"  [OK] language detection changed: {a_lang} -> {b_lang}")
        else:
            print(f"  [WARN] language unchanged ({a_lang}) -- trim may not have helped, "
                  "or both windows already had speech")

        if args.keep_trimmed:
            print(f"\n[output] trimmed file kept at: {trimmed}")

        print("\nTune hints:")
        print("  - trim cut too much (real opening missing): "
              "lower --duration (e.g. 1.0) or --threshold (e.g. -40)")
        print("  - trim cut too little (still garble in BEFORE-trim transcript): "
              "raise --duration (e.g. 3.0) or --threshold (e.g. -30)")
        print("  - second run: append --keep-trimmed to inspect the trimmed file in a player")
    finally:
        if not args.keep_trimmed and trimmed.exists():
            trimmed.unlink()


if __name__ == "__main__":
    main()
