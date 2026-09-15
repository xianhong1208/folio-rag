#!/usr/bin/env python3
"""Test openai-whisper with anti-hallucination decode params -- NO download needed.

Uses the existing local model at assets/whisper_models/large-v3-turbo.pt to bypass
the slow faster-whisper CT2 download. If tuned decode params alone fix the
hallucination, we don't need faster-whisper at all -- just override Docling's
call (subclass + monkey-patch _NativeWhisperModel.transcribe).

Two configs (default: runs both):
  A. VANILLA -- just verbose + word_timestamps (matches what Docling does)
  B. TUNED   -- + language=zh, condition_on_previous_text=False, no_speech=0.85,
                logprob=-1.0, compression_ratio=2.0, initial_prompt

Usage:
    uv run --group cuda python scripts/test_whisper_tuned.py /path/to/audio.mp3
    uv run --group cuda python scripts/test_whisper_tuned.py /path/to/audio.mp3 \\
        --device cuda:4 --skip-vanilla
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def check_hallucinations(text: str):
    hits = []
    for pat, desc in _KNOWN_HALLUCINATIONS:
        m = re.search(pat, text)
        if m:
            hits.append((m.group(0), desc))
    return hits


def load_local_whisper(device: str):
    import whisper
    whisper_dir = PROJECT_ROOT / "assets" / "whisper_models"
    if not (whisper_dir.is_dir() and any(whisper_dir.glob("*.pt"))):
        sys.exit(f"[error] no .pt found in {whisper_dir} -- expected local model file")
    print(f"[whisper] loading turbo from {whisper_dir} (device={device})")
    return whisper.load_model("turbo", device=device, download_root=str(whisper_dir))


def run_pass(model, audio_path: Path, *, tuned: bool, label: str, device: str):
    print(f"\n========== {label} ==========")
    t0 = time.time()

    kwargs = {
        "verbose": False,
        "word_timestamps": True,
        "fp16": (device != "cpu"),
    }
    if tuned:
        kwargs.update({
            "language": "zh",
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.85,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.0,
            "initial_prompt": "這是繁體中文的會議錄音轉錄。",
        })
        print("  tuned params:")
        print("    language=zh  condition_on_previous_text=False")
        print("    no_speech=0.85  logprob=-1.0  compression_ratio=2.0")
        print("    initial_prompt='這是繁體中文的會議錄音轉錄。'")
    else:
        print("  vanilla (matches Docling's call: only verbose + word_timestamps)")

    result = model.transcribe(str(audio_path), **kwargs)
    elapsed = time.time() - t0

    text = result.get("text", "")
    segments = result.get("segments", [])
    language = result.get("language", "?")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  language detected: {language}")
    print(f"  total segments: {len(segments)}")
    print(f"  total chars: {len(text)}")
    return elapsed, text, segments


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--device", default="cuda:0",
                        help="cuda / cpu / cuda:N (default: cuda:0)")
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--skip-vanilla", action="store_true",
                        help="skip vanilla control, only run tuned")
    args = parser.parse_args()

    src = args.input.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"[error] file not found: {src}")

    try:
        import whisper  # openai-whisper
    except ImportError:
        sys.exit("[error] openai-whisper not installed (should be via docling[asr])")

    print(f"[input]  {src}")
    print(f"[device] {args.device}")

    t0 = time.time()
    model = load_local_whisper(args.device)
    print(f"[model]  loaded in {time.time() - t0:.1f}s")

    runs = []

    if not args.skip_vanilla:
        a_elapsed, a_text, a_segments = run_pass(
            model, src, tuned=False, label="A: VANILLA (matches Docling)",
            device=args.device,
        )
        runs.append(("A_VANILLA", a_elapsed, a_text, a_segments))

    b_elapsed, b_text, b_segments = run_pass(
        model, src, tuned=True, label="B: TUNED (anti-hallucination params)",
        device=args.device,
    )
    runs.append(("B_TUNED", b_elapsed, b_text, b_segments))

    args.save_dir.mkdir(exist_ok=True, parents=True)

    print("\n========== TAIL OF TRANSCRIPT (last 5 segments) ==========")
    for label, _elapsed, text, segments in runs:
        print(f"\n--- {label} ---")
        for s in segments[-5:]:
            preview = s["text"].strip()[:80]
            print(f"  [{s['start']:7.2f}s --> {s['end']:7.2f}s]  {preview}")
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
        if a_hits and not b_hits:
            print("  [WIN]  Tuned decode params eliminated the hallucinations.")
            print("         You DON'T need faster-whisper -- can fix this in current pipeline")
            print("         by subclassing _NativeWhisperModel.transcribe() to pass these params.")
        elif not a_hits and not b_hits:
            print("  [INFO] Neither has hallucinations.")
            print("         Either this audio doesn't trigger them, or trim+language already fixed it.")
        elif a_hits and b_hits:
            print("  [BAD]  Params alone don't fix it -- need VAD (faster-whisper) or Chinese fine-tune.")
        else:
            print("  [???]  Tuning introduced hallucinations? Inspect transcripts.")
    else:
        b_hits = check_hallucinations(runs[0][2])
        if b_hits:
            print(f"  [BAD]  Tuned still has {len(b_hits)} hallucination(s) -- need VAD next.")
        else:
            print("  [WIN]  Tuned is clean. Recommend overriding Docling's whisper call.")


if __name__ == "__main__":
    main()
