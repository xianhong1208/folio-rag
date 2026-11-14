"""``main.py backup`` and ``main.py restore`` (see docs/backup.md).

Exit codes: 0 done, 1 failed, 2 refused (a precondition: running index jobs, a
non-empty target, a version or model mismatch, a damaged archive).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def _say(msg: str) -> None:
    print(msg, flush=True)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Folio backup and restore")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backup", help="write one archive: the database, storage/ and a manifest")
    b.add_argument("--config-dir", default="config")
    where = b.add_mutually_exclusive_group()
    where.add_argument("--output", type=Path, help="archive file to write (default: "
                       "<database.backup.dir>/folio-backup-<UTC time>.tar)")
    where.add_argument("--dir", type=Path, help="directory for a timestamped archive "
                       "(default: database.backup.dir)")
    b.add_argument("--keep", type=int, help="afterwards, keep only the newest N timestamped "
                   "archives in that directory")
    b.add_argument("--allow-running-jobs", action="store_true",
                   help="back up even while index jobs run (they are marked interrupted after a restore)")

    r = sub.add_parser("restore", help="restore an archive into the configured database and storage/")
    r.add_argument("archive", type=Path)
    r.add_argument("--config-dir", default="config")
    r.add_argument("--check", action="store_true",
                   help="only verify the archive and compare it with this Folio; change nothing")
    r.add_argument("--force", action="store_true",
                   help="replace a database that has tables (dropped) and a storage/ that has files "
                        "(moved aside to storage.before-restore-<time>)")
    r.add_argument("--allow-mismatch", action="store_true",
                   help="restore although the archive comes from a newer Folio or another embedding "
                        "model / dimension")
    return p


def _load_config(config_dir: str):
    from src.config.config_manager import Config
    from src.log import setup_logger
    Config.set_config(config_dir)
    model = Config.get_config_model()
    if model is None:
        raise RuntimeError(f"the configuration in {config_dir}/ is invalid (see the error above)")
    log_conf = Config.get_logging_config()
    # Details go to the log files; the terminal only gets this command's own lines.
    setup_logger(console_level="ERROR", file_level="DEBUG",
                 log_base_dir=getattr(log_conf, "log_dir", "logs") if log_conf else "logs")
    return model


def _backup(args) -> int:
    from src.backup import service
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    model = _load_config(args.config_dir)
    conf = Config.get_database_config().backup
    if args.keep is not None and args.keep < 1:
        _say("--keep must be at least 1")
        return EXIT_REFUSED
    out_dir = args.dir or service.backup_dir(conf)
    result = service.create_backup(
        db_url=resolve_db_url(Config.get_database_config()), storage_root=service.default_storage_root(),
        out_dir=None if args.output else out_dir, out_path=args.output,
        allow_running_jobs=args.allow_running_jobs, bin_dir=conf.pg_bin_dir,
        rag_conf=getattr(model, "rag", None), log=_say)
    m = result.manifest
    chunks = sum(f["chunks"] for f in m["folders"])
    _say(f"wrote {result.path} ({result.size:,} bytes): {len(m['folders'])} folder(s), {chunks:,} chunks, "
         f"{m['storage']['files']} stored file(s)")
    for w in result.warnings:
        _say(f"warning: {w}")
    if args.keep:
        for p in service.prune(result.path.parent, args.keep):
            _say(f"removed old archive {p.name}")
    return EXIT_OK


def _restore(args) -> int:
    from src.backup import service
    from src.config.config_manager import Config
    from src.utils.db_url import resolve_db_url
    model = _load_config(args.config_dir)
    conf = Config.get_database_config().backup
    result = service.restore_backup(
        args.archive, db_url=resolve_db_url(Config.get_database_config()),
        storage_root=service.default_storage_root(), force=args.force,
        allow_mismatch=args.allow_mismatch, check_only=args.check, bin_dir=conf.pg_bin_dir,
        rag_conf=getattr(model, "rag", None), log=_say)
    _say(f"archive: {service.manifest_summary(result.manifest)}")
    for w in result.warnings:
        _say(f"warning: {w}")
    if not result.restored:
        _say("the archive is intact and can be restored into this Folio")
        return EXIT_OK
    for f in result.folders:
        _say(f"restored folder {f['id']} {f['name']!r}: {f['chunks']:,} chunks")
    _say("done. Start Folio: it migrates the database if the backup came from an older version.")
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    from src.backup.archive import ArchiveError
    from src.backup.pg import PgToolError
    from src.backup.service import BackupRefused
    try:
        return _backup(args) if args.command == "backup" else _restore(args)
    except (BackupRefused, ArchiveError) as e:
        _say(f"refused: {e}")
        return EXIT_REFUSED
    except PgToolError as e:
        _say(f"failed: {e}")
        return EXIT_FAILED
    except Exception as e:  # noqa: BLE001
        _say(f"failed: {type(e).__name__}: {e}")
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
