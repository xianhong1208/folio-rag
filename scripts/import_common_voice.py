
"""Import Common Voice (zh-TW) clips as ASR evaluation material.

Samples N validated clips from a downloaded Common Voice tar.gz and lands them
in this repo's evaluation format:
    evals/asr/audio/cv_<clip_name>.mp3
    evals/asr/golden_transcripts/cv_<clip_name>.txt   <- human-validated
                                                         sentence, no proofreading

Usage:
    # Mozilla Data Collective bundle (tsv inside the tar):
    uv run --no-sync python scripts/import_common_voice.py <tar.gz path> [--n 200] [--seed 42]
    # HF mirror (fsicoli/common_voice_*: audio shard .tar + transcript .tsv separate):
    uv run --no-sync python scripts/import_common_voice.py <shard.tar> --tsv <test.tsv>

- Fixed sampling seed for reproducibility; a rerun first clears old cv_* files.
- CC0-1.0 (public domain); the material stays out of the repo (gitignored).
- Only validated (human-verified) clips are used; the TSV is auto-detected
  (a validated*.tsv with path + sentence columns wins, else the first match).
"""

import argparse
import csv
import random
import sys
import tarfile
from pathlib import Path


def find_tsv(tar: tarfile.TarFile):
    """Pick the transcript TSV (validated preferred); return (member, columns)."""
    tsvs = [m for m in tar.getmembers() if m.name.endswith(".tsv")]
    tsvs.sort(key=lambda m: (0 if "validated" in Path(m.name).name else 1, m.name))
    for m in tsvs:
        f = tar.extractfile(m)
        if f is None:
            continue
        head = f.readline().decode("utf-8", errors="replace")
        cols = head.rstrip("\n").split("\t")
        if "path" in cols and "sentence" in cols:
            return m, cols
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tarball", help="path to the Common Voice tar(.gz)")
    ap.add_argument("--tsv", help="external transcript TSV (for HF mirrors that split audio/transcript)")
    ap.add_argument("--n", type=int, default=200, help="number of clips to sample (default: 200)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    audio_dir = Path("evals/asr/audio")
    golden_dir = Path("evals/asr/golden_transcripts")
    golden_dir.mkdir(parents=True, exist_ok=True)

    # Clear old cv_* samples so a re-sample does not mix batches
    for old in list(audio_dir.glob("cv_*")) + list(golden_dir.glob("cv_*")):
        old.unlink()

    with tarfile.open(args.tarball, "r:*") as tar:
        if args.tsv:
            lines = Path(args.tsv).read_text(encoding="utf-8").splitlines()
            reader = csv.DictReader(lines, delimiter="\t")
            print(f"transcript table: {args.tsv} (external)")
        else:
            tsv_member, _ = find_tsv(tar)
            if tsv_member is None:
                print("no .tsv with path+sentence columns found in the tar (pass --tsv for HF mirrors)")
                return 2
            print(f"transcript table: {tsv_member.name}")
            f = tar.extractfile(tsv_member)
            reader = csv.DictReader(
                (line.decode("utf-8", errors="replace") for line in f), delimiter="\t")

        rows = []
        for row in reader:
            s = (row.get("sentence") or "").strip()
            p = (row.get("path") or "").strip()
            if s and p:
                rows.append((p, s))
        print(f"available clips: {len(rows)}")
        if not rows:
            return 2

        random.Random(args.seed).shuffle(rows)
        picked = rows[: args.n]

        # Index clip members by file name (bundle keeps them under */clips/,
        # HF shard lays them flat under <shard>/)
        by_name = {}
        for m in tar.getmembers():
            if m.isfile() and Path(m.name).suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}:
                by_name[Path(m.name).name] = m

        done = 0
        for p, s in picked:
            m = by_name.get(Path(p).name)
            if m is None:
                continue
            stem = "cv_" + Path(p).stem
            with open(audio_dir / (stem + Path(p).suffix), "wb") as out:
                out.write(tar.extractfile(m).read())
            (golden_dir / (stem + ".txt")).write_text(s + "\n", encoding="utf-8")
            done += 1

    print(f"wrote {done}/{len(picked)} clips → evals/asr/audio/ + golden_transcripts/")
    print("next: uv run --no-sync python scripts/run_asr_eval.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
