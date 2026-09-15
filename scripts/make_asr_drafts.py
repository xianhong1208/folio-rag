
"""Generate ASR golden-transcript drafts.

Typing full golden transcripts from scratch is impractical, so ASR produces
drafts into evals/asr/drafts/; a human proofreads against the audio and moves
the result into golden_transcripts/ (workflow in evals/README.md).

Usage:
    uv run --no-sync python scripts/make_asr_drafts.py                  # whisper (default)
    uv run --no-sync python scripts/make_asr_drafts.py --provider fireredasr
        # -> drafts/<stem>.fireredasr.txt (a second opinion to compare side by
        #    side with the whisper draft; passages where the two disagree are
        #    the ones to listen to first when proofreading)

Warning: drafts/ is machine output and must never be moved straight into
golden_transcripts/ for scoring (CER would then measure against the draft
model's own errors).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="docling-whisper",
                    choices=["docling-whisper", "fireredasr"])
    ap.add_argument("--force", action="store_true", help="Overwrite existing drafts")
    args = ap.parse_args()

    audio_dir = Path("evals/asr/audio")
    out_dir = Path("evals/asr/drafts")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in audio_dir.iterdir()
                   if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        print("evals/asr/audio/ has no audio files — add source material first (see evals/README.md)")
        return 2

    from src.config.model import AsrConfig
    from src.domain.rag.asr_provider import create_asr_provider

    provider = create_asr_provider(AsrConfig(provider=args.provider))
    if not provider.available():
        print(f"provider '{args.provider}' unavailable (model/weights not ready)")
        return 2

    # The whisper draft is primary (no suffix); other providers get a suffix as a second opinion
    suffix = "" if args.provider == "docling-whisper" else f".{args.provider}"

    for f in files:
        dst = out_dir / f"{f.stem}{suffix}.txt"
        if dst.exists() and not args.force:
            print(f"skip (exists, use --force to overwrite): {dst.name}")
            continue
        t0 = time.time()
        r = provider.transcribe(audio_path=str(f), file_name=f.name)
        dst.write_text(r.text.strip() + "\n", encoding="utf-8")
        print(f"{f.name}: {len(r.text)} chars, {time.time() - t0:.0f}s → {dst.name}")

    print("\nDrafts complete — proofread by hand, then move into evals/asr/golden_transcripts/ (drop the suffix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
