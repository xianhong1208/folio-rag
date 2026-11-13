"""Convert a Folio 1.4 config/config.yaml into the six config files of 1.5 plus .env lines.

    uv run python scripts/split_config.py                      # config/config.yaml -> config/
    uv run python scripts/split_config.py --force              # replace the six template files
    uv run python scripts/split_config.py --old old.yaml --out-dir /tmp/cfg --env-out /tmp/cfg/env

Writes database.yaml, auth.yaml, models.yaml, mcp.yaml, retrieval.yaml and system.yaml to
--out-dir, and the per-machine / secret values to --env-out (mode 600) as KEY=value lines
to merge into .env. Values are never printed: the output lists file names, variable names
and notes only. Afterwards it loads both configurations and compares them field by field
(--no-check skips that); it exits 1 if they differ.

The environment used for ${VAR} references is this process's environment plus --env-file
(default: the project's .env, as the server loads it).
See docs/configuration.md, "Upgrading from 1.4".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.config.dotenv import read_dotenv  # noqa: E402
from src.config.split import check, split_legacy, write_outputs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", default="config/config.yaml", help="the 1.4 config file to convert")
    ap.add_argument("--out-dir", default="config", help="where the six files are written")
    ap.add_argument("--env-out", default=".env.split",
                    help="file for the .env lines (mode 600; merge into .env, then delete)")
    ap.add_argument("--env-file", default=str(ROOT / ".env"),
                    help="the .env the server reads (used to expand ${VAR}; '' = none)")
    ap.add_argument("--force", action="store_true", help="replace existing output files")
    ap.add_argument("--no-check", action="store_true", help="skip the field-by-field comparison")
    args = ap.parse_args(argv)

    old_path = Path(args.old)
    if not old_path.is_file():
        print(f"error: {old_path} not found", file=sys.stderr)
        return 2
    old_raw = yaml.safe_load(old_path.read_text(encoding="utf-8")) or {}
    env = dict(os.environ)
    if args.env_file:
        env.update(read_dotenv(args.env_file))

    result = split_legacy(old_raw, env)
    out_dir, env_out = Path(args.out_dir), Path(args.env_out)
    try:
        write_outputs(result, out_dir, env_out, force=args.force)
    except (FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"Wrote the six config files to {out_dir}/")
    if result.env_lines:
        print(f"Wrote {len(result.env_lines)} .env line(s) to {env_out} (mode 600): "
              + ", ".join(sorted(result.env_lines)))
        print("  Merge them into .env (replace any line with the same name), then delete the file.")
    else:
        print(f"No .env lines needed ({env_out} holds only comments).")
    for note in result.notes:
        print(f"note: {note}")

    if args.no_check:
        return 0
    diffs = check(old_raw, out_dir, env, result.env_lines)
    if diffs:
        print("CHECK FAILED — fields that differ from the old config: " + ", ".join(diffs))
        return 1
    print("Check passed: the converted configuration equals the old one field by field "
          "(rag.answer_llm / rag.eval_llm and module wiring excepted).")
    if old_path.resolve().parent == out_dir.resolve():
        print(f"Now move {old_path} out of {out_dir}/: it is no longer read, and startup "
              "reports it until it is gone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
