#!/usr/bin/env python3
"""Test faster-whisper to see if it eliminates Chinese hallucinations.

Runs two passes on the same audio (full file, since hallucinations appear at the tail):
  A. NO VAD       — closer to openai-whisper / Docling default behavior
  B. VAD enabled  — recommended config (Silero VAD skips silent regions entirely)

Prints last 5 segments of each + checks for known Chinese Whisper hallucinations
(YouTube training-data leakage like "请不吝点赞...明镜与点点").

Usage:
    uv run python scripts/test_faster_whisper.py /path/to/audio.mp3
    uv run python scripts/test_faster_whisper.py /path/to/audio.mp3 \\
        --device cpu --model large-v3
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Known Chinese hallucination patterns — YouTube channel outros / fan-subbing credits
# that bled into Whisper's training data. If ANY of these appear, it's not in your audio.
_KNOWN_HALLUCINATIONS = [
    (r"请不吝点[赞讚]", "明镜与点点 outro (simplified)"),
    (r"請不吝點[赞讚]", "明鏡與點點 outro (traditional)"),
    (r"明[镜鏡][与與]点[点點]", "明镜与点点 channel name"),
    (r"订阅.*[频頻]道", "generic subscribe-channel"),
    (r"訂閱.*[频頻]道", "generic subscribe-channel (trad)"),
    (r"打[赏賞]支持", "打赏支持"),
    (r"字幕由.*Amara", "Amara.org community subtitles"),
    (r"由志愿者.*提供", "by volunteers"),
    (r"字幕志[愿願]者", "subtitle volunteers"),
    (r"感谢观看", "thanks for watching"),
    (r"感謝觀看", "thanks for watching (trad)"),
]


def check_hallucinations(text: str) -> list[tuple[str, str]]:
    """Return list of (matched_substring, description) for known hallucinations found."""
    hits = []
    for pat, desc in _KNOWN_HALLUCINATIONS:
        m = re.search(pat, text)
        if m:
            hits.append((m.group(0), desc))
    return hits


def run_pass(model, audio_path: Path, *, vad: bool, label: str):
    print(f"\n========== {label} ==========")
    print(f"  vad_filter={vad}  condition_on_previous_text=False  language=zh")
    t0 = time.time()

    kwargs = {
        "language": "zh",
        "condition_on_previous_text": False,
    }
    if vad:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=2000)

    segments_iter, info = model.transcribe(str(audio_path), **kwargs)
    segments = list(segments_iter)  # generator -> list to allow indexing
    elapsed = time.time() - t0

    full_text = "".join(s.text for s in segments)
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  language detected: {info.language} ({info.language_probability:.1%})")
    print(f"  total segments: {len(segments)}")
    print(f"  total chars: {len(full_text)}")
    return elapsed, full_text, segments


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Audio file (.mp3/.wav)")
    parser.add_argument("--model", default="large-v3-turbo",
                        help="model size (default: large-v3-turbo). "
                             "Try 'large-v3' if turbo CT2 weights unavailable.")
    parser.add_argument("--device", default="cuda",
                        help="cuda / cpu / cuda:N (e.g. cuda:4 for 5th GPU). Default: cuda")
    parser.add_argument("--device-index", type=int, default=0,
                        help="GPU index, overridden if --device uses cuda:N notation")
    parser.add_argument("--compute-type", default=None,
                        help="default: float16 on cuda, int8 on cpu")
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp"),
                        help="dir to write full transcripts (default: /tmp)")
    parser.add_argument("--skip-novad", action="store_true",
                        help="only run VAD pass, skip the no-VAD control")
    args = parser.parse_args()

    src = args.input.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"[error] file not found: {src}")

    # Parse `cuda:N` shorthand → device="cuda", device_index=N
    device = args.device
    device_index = args.device_index
    if device.startswith("cuda:"):
        try:
            device_index = int(device.split(":", 1)[1])
            device = "cuda"
        except ValueError:
            sys.exit(f"[error] invalid device spec: '{args.device}' (expected cuda / cpu / cuda:N)")
    elif device not in ("cuda", "cpu"):
        sys.exit(f"[error] invalid device: '{device}' (expected cuda / cpu / cuda:N)")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("[error] faster-whisper not installed. Run: uv add faster-whisper")

    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")

    print(f"[input]  {src}")
    print(f"[model]  {args.model} on {device}:{device_index} ({compute_type})")
    print(f"[note]   first run downloads CT2 weights from HuggingFace (~1.5GB) -- be patient")

    t0 = time.time()
    model = WhisperModel(
        args.model,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )
    print(f"[model]  loaded in {time.time() - t0:.1f}s")

    runs = []

    if not args.skip_novad:
        a_elapsed, a_text, a_segments = run_pass(
            model, src, vad=False, label="A: NO VAD (Docling-equivalent default)"
        )
        runs.append(("A_NOVAD", a_elapsed, a_text, a_segments))

    b_elapsed, b_text, b_segments = run_pass(
        model, src, vad=True, label="B: VAD ENABLED (recommended)"
    )
    runs.append(("B_VAD", b_elapsed, b_text, b_segments))

    args.save_dir.mkdir(exist_ok=True, parents=True)

    print("\n========== TAIL OF TRANSCRIPT (last 5 segments) ==========")
    for label, _elapsed, text, segments in runs:
        print(f"\n--- {label} ---")
        for s in segments[-5:]:
            preview = s.text.strip()[:80]
            print(f"  [{s.start:7.2f}s --> {s.end:7.2f}s]  {preview}")
        out_path = args.save_dir / f"{src.stem}_{label}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  full transcript saved: {out_path}")

    print("\n========== HALLUCINATION CHECK ==========")
    for label, _elapsed, text, _segments in runs:
        hits = check_hallucinations(text)
        if hits:
            print(f"  {label}: {len(hits)} match(es)")
            for matched, desc in hits:
                print(f"    - '{matched}' ({desc})")
        else:
            print(f"  {label}: clean (no known hallucinations)")

    print("\n========== VERDICT ==========")
    if len(runs) == 2:
        a_hits = check_hallucinations(runs[0][2])
        b_hits = check_hallucinations(runs[1][2])
        a_elapsed = runs[0][1]
        b_elapsed = runs[1][1]

        if a_hits and not b_hits:
            print("  [WIN]  VAD eliminated the hallucinations.")
            print("         Recommend: switch RAG indexer to use faster-whisper with VAD")
        elif not a_hits and not b_hits:
            print("  [INFO] Neither run has known hallucinations.")
            print("         Either faster-whisper's defaults are better than openai-whisper's,")
            print("         OR your audio doesn't trigger the YouTube-outro pattern.")
        elif a_hits and b_hits:
            print("  [BAD]  Hallucinations persist even with VAD.")
            print("         Next try: BELLE-2/Belle-whisper-large-v3-turbo-zh (Chinese fine-tune)")
        else:
            print("  [???]  VAD introduced hallucinations? Inspect transcripts.")

        speedup = (a_elapsed / b_elapsed) if b_elapsed else 0
        print(f"\n  Speed: A={a_elapsed:.0f}s  B={b_elapsed:.0f}s  (VAD is {speedup:.1f}x faster)")
    else:
        b_hits = check_hallucinations(runs[0][2])
        if b_hits:
            print(f"  [BAD]  VAD pass still has {len(b_hits)} hallucination(s). "
                  "Try Chinese fine-tuned variant next.")
        else:
            print("  [WIN]  VAD pass is clean. Recommend switching RAG indexer to faster-whisper.")


if __name__ == "__main__":
    main()
