
"""Score ASR evaluation: run golden transcripts through the configured endpoint and produce a CER table.

Usage:
    uv run --no-sync python scripts/run_asr_eval.py                 # endpoint and model from rag.asr
    uv run --no-sync python scripts/run_asr_eval.py --model whisper-large-v3-turbo
    uv run --no-sync python scripts/run_asr_eval.py --baseline evals/baselines/asr_xxx.json

The endpoint is rag.asr's OpenAI-compatible one (ASR_BASE_URL / ASR_API_KEY / ASR_MODEL);
there is no in-process speech-to-text since 1.5.

Material (see evals/README.md):
    evals/asr/audio/<name>.{wav,mp3,m4a,aac,ogg,flac}
    evals/asr/golden_transcripts/<name>.txt     # human-proofread Traditional-Chinese transcript

Metrics: CER (character error rate, punctuation/whitespace stripped), simplified
-character ratio (~0 for valid Traditional Chinese), hallucination-pattern match
count (audio_defense), and elapsed time.
Evaluated on the provider's raw output (no leading-silence trimming or
hallucination filtering — those are the production defense layer; here the
hallucination count is reported as the filter's match count for cross-model
comparison).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

AUDIO_DIR = REPO / "evals" / "asr" / "audio"
GOLDEN_DIR = REPO / "evals" / "asr" / "golden_transcripts"
BASELINE_DIR = REPO / "evals" / "baselines"
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override rag.asr.model from config")
    ap.add_argument("--baseline", help="compare against an existing baseline JSON")
    ap.add_argument("--config-dir", default="config",
                    help="directory with the six config files (secrets come from .env)")
    args = ap.parse_args()

    pairs = []
    for f in sorted(AUDIO_DIR.iterdir()) if AUDIO_DIR.is_dir() else []:
        if f.suffix.lower() in _AUDIO_EXTS:
            g = GOLDEN_DIR / f"{f.stem}.txt"
            if g.is_file():
                pairs.append((f, g))
            else:
                print(f"WARNING: {f.name} has no matching golden_transcripts/{f.stem}.txt — skipping")
    if not pairs:
        print("No eval material. Please provide:")
        print(f"   {AUDIO_DIR.relative_to(REPO)}/<name>.wav|mp3|m4a|aac|ogg|flac")
        print(f"   {GOLDEN_DIR.relative_to(REPO)}/<name>.txt (human-proofread Traditional-Chinese transcript)")
        return 2

    from src.config.dotenv import load_dotenv
    load_dotenv()
    from src.config.config_manager import Config
    Config.set_config(args.config_dir)
    from src.config.model import AsrConfig
    from src.domain.rag.asr_provider import create_asr_provider
    from src.domain.rag.audio_defense import _filter_whisper_hallucinations
    from evals.metrics import cer, simplified_char_ratio

    asr_cfg = getattr(getattr(Config.get_config_model(), "rag", None), "asr", None)
    if args.model:
        base = asr_cfg.model_dump() if asr_cfg else {}
        base["model"] = args.model
        asr_cfg = AsrConfig(**base)
    provider = create_asr_provider(asr_cfg)
    provider_name = (asr_cfg.model if asr_cfg and asr_cfg.model else "openai-compatible").replace("/", "_")
    if not provider.available():
        print("speech-to-text not available: set rag.asr.base_url (ASR_BASE_URL) and enable rag.asr")
        return 2

    rows = []
    for audio, golden in pairs:
        ref = golden.read_text(encoding="utf-8")
        t0 = time.perf_counter()
        result = provider.transcribe(audio_path=str(audio), file_name=audio.name)
        elapsed = time.perf_counter() - t0
        _, n_halluc = _filter_whisper_hallucinations(result.text)
        rows.append({
            "file": audio.name,
            "cer": round(cer(ref, result.text), 4),
            "simplified_ratio": round(simplified_char_ratio(result.text), 4),
            "hallucination_matches": n_halluc,
            "elapsed_s": round(elapsed, 1),
            "ref_chars": len(ref),
            "hyp_chars": len(result.text),
            # Store full text in the baseline so error attribution can be done
            # later (e.g. recompute CER with s2twp appended, to separate
            # simplified/traditional glyph differences from real character
            # errors) without rerunning transcription
            "ref": ref.strip(),
            "hyp": result.text.strip(),
        })
        print(f"  {audio.name}: CER={rows[-1]['cer']:.2%} simplified={rows[-1]['simplified_ratio']:.2%} "
              f"hallucinations={n_halluc} {elapsed:.1f}s")

    summary = {
        "provider": provider_name,
        "date": date.today().isoformat(),
        "avg_cer": round(sum(r["cer"] for r in rows) / len(rows), 4),
        "avg_simplified_ratio": round(sum(r["simplified_ratio"] for r in rows) / len(rows), 4),
        "total_hallucinations": sum(r["hallucination_matches"] for r in rows),
        "files": rows,
    }
    out = BASELINE_DIR / f"asr_{provider_name}_{date.today().isoformat()}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== {provider_name} ==  avg CER {summary['avg_cer']:.2%} | "
          f"simplified {summary['avg_simplified_ratio']:.2%} | hallucinations {summary['total_hallucinations']}")
    print(f"baseline written: {out.relative_to(REPO)}")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        d = summary["avg_cer"] - base["avg_cer"]
        print(f"vs {base['provider']} ({base['date']}): CER {base['avg_cer']:.2%} → "
              f"{summary['avg_cer']:.2%} ({'+' if d >= 0 else ''}{d:.2%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
