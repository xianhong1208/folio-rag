
"""Generate ASR golden-transcript drafts.

Typing full golden transcripts from scratch is impractical, so ASR produces
drafts into evals/asr/drafts/; a human proofreads against the audio and moves
the result into golden_transcripts/ (workflow in evals/README.md).

Usage:
    uv run --no-sync python scripts/make_asr_drafts.py                  # rag.asr's endpoint
    uv run --no-sync python scripts/make_asr_drafts.py --model other-model --suffix other
        # -> drafts/<stem>.other.txt (a second opinion to compare side by side
        #    with the first draft; passages where the two disagree are the ones
        #    to listen to first when proofreading)

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
    ap.add_argument("--model", help="override rag.asr.model from config")
    ap.add_argument("--suffix", default="", help="draft name suffix for a second opinion")
    ap.add_argument("--config-dir", default="config")
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

    from src.config.dotenv import load_dotenv
    load_dotenv()
    from src.config.config_manager import Config
    Config.set_config(args.config_dir)
    from src.config.model import AsrConfig
    from src.domain.rag.asr_provider import create_asr_provider

    asr_cfg = getattr(getattr(Config.get_config_model(), "rag", None), "asr", None)
    if args.model:
        base = asr_cfg.model_dump() if asr_cfg else {}
        base["model"] = args.model
        asr_cfg = AsrConfig(**base)
    provider = create_asr_provider(asr_cfg)
    if not provider.available():
        print("speech-to-text not available: set rag.asr.base_url (ASR_BASE_URL) and enable rag.asr")
        return 2

    # The first draft has no suffix; a second opinion gets one
    suffix = f".{args.suffix}" if args.suffix else ""

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
