"""Folio MCP Server CLI entry point.

Default run (`uv run python main.py`) does DB bootstrap, auto migration, Docling
warmup (only when a role uses docling's local models), then starts the FastAPI +
FastMCP server. See main() for the CLI flags.

Note: this file deliberately does not use `from __future__ import annotations`.
Nuitka's libs-loader injection prepends executable code to the top of main.py,
which would violate the rule that `from __future__` must precede all executable
code. All type hints use builtin types, so PEP 563 is not needed here.
"""

import argparse
import sys
from pathlib import Path

# Load the project's .env before anything reads os.environ. config/*.yaml expands
# ${DATABASE_URL:-...}, ${LLM_BASE_URL:-...} and the other per-machine values, so they
# must be present first. override=True lets this project's .env win over a stale global
# export lingering in the shell.
from src.config.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv(Path(__file__).resolve().parent / ".env")

# Let the PyTorch CUDA allocator grow segments instead of fragmenting them: on a
# card shared with vLLM, fragmentation alone causes out-of-memory errors in the
# docling stages. Must be set before torch is imported; a user value wins.
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Disable core dumps for this process by default. At interpreter shutdown a native
# library aborts (SIGABRT with si_code=SI_TKILL, i.e. abort() from inside the
# process — torch/CUDA-style teardown) and, with the launcher's `ulimit -c
# unlimited`, that left a ~16 GB core.<pid> in the repo root on every restart.
# Set FOLIO_CORE_DUMPS=1 (formerly AGENTIC_RAG_CORE_DUMPS) to keep the launcher's limit when a native
# backtrace is actually wanted.
def _disable_core_dumps() -> None:
    from src.utils import env_compat
    if (env_compat.getenv("FOLIO_CORE_DUMPS") or "").strip().lower() in ("1", "true", "yes"):
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass  # non-POSIX platform or not permitted — nothing to do


_disable_core_dumps()

import uvicorn

from app import create_app
from db import db as db_module
from db.migrate import DatabaseMigrator
from src.config.config_manager import Config, ConfigFileError
from src.log import get_server_logger, setup_logger


def run_migration_command(action: str, message: str = None, revision: str = "-1") -> bool:
    """Execute an Alembic migration action.

    Actions:
        auto / migrate     : Auto-apply all pending migrations (default at startup).
        upgrade            : Upgrade to a given revision (--revision, default head).
        downgrade          : Downgrade to a given revision.
        rollback           : Roll back one revision (same as downgrade -1).
        generate           : Generate a new migration (requires --message).
        status / current   : Check the current status.
        history / heads    : Show history / all heads.
        init               : Create the database (if absent) and apply migrations.
        stamp              : Stamp a revision without executing it.
    """
    logger = get_server_logger()
    try:
        migrator = DatabaseMigrator()

        actions = {
            "auto":      lambda: migrator.auto_migrate(),
            "status":    lambda: (migrator.check_status(), True)[1],
            "migrate":   lambda: migrator.apply_migrations(),
            "upgrade":   lambda: migrator.upgrade(revision if revision != "-1" else "head"),
            "rollback":  lambda: migrator.rollback_migration(revision),
            "downgrade": lambda: migrator.downgrade(revision),
            "init":      lambda: migrator.initialize_database(),
            "stamp":     lambda: migrator.stamp(revision if revision != "-1" else "head"),
            "current":   lambda: migrator.show_current_revision(),
            "history":   lambda: migrator.show_migration_history(),
            "heads":     lambda: migrator.show_heads(),
        }

        if action == "generate":
            if not message:
                logger.error("❌ Migration message is required for generate (use -m)")
                return False
            return migrator.create_migration(message)

        handler = actions.get(action)
        if handler is None:
            valid = ", ".join(sorted(list(actions.keys()) + ["generate"]))
            logger.error(f"❌ Unknown migration action: '{action}'. Valid: {valid}")
            return False

        return handler()

    except Exception as e:
        logger.error(f"❌ Migration error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Folio MCP Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", type=str, default="config",
                        help="Directory holding the six config files (database.yaml, auth.yaml, "
                             "models.yaml, mcp.yaml, retrieval.yaml, system.yaml)")
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", type=str,
                       choices=["critical", "error", "warning", "info", "debug", "trace"])
    parser.add_argument("--transport", type=str, choices=["sse", "http"])

    parser.add_argument("--no-migrate", action="store_true",
                       help="Skip auto-migration on startup (default: auto-migrate runs)")
    parser.add_argument("--migrate-only", nargs="?", const="auto", metavar="ACTION",
                       help="Run migration only, don't start server. Actions: auto, status, "
                            "migrate, upgrade, rollback, downgrade, generate, init, stamp, "
                            "current, history, heads")
    parser.add_argument("--message", "-m", help="Migration message (for --migrate-only generate)")
    parser.add_argument("--revision", "-r", default="-1",
                       help="Target revision for upgrade/rollback/stamp (default: -1)")

    args = parser.parse_args()

    try:
        Config.set_config(args.config_dir)
        config_obj = Config.get_config_model()
        if not config_obj:
            raise RuntimeError("the configuration is invalid (see the error above)")

        log_conf = Config.get_logging_config()
        if not log_conf:
            raise RuntimeError("Logging config missing")

        log_level = str(getattr(log_conf, "level", "INFO")).upper()
        if args.log_level:
            log_level = args.log_level.upper()
        log_dir = getattr(log_conf, "log_dir", "logs")

        # logging.max_size / backup_count: rotate at that size, keep that many old files.
        setup_logger(console_level=log_level, file_level="DEBUG", log_base_dir=log_dir,
                     rotation=getattr(log_conf, "max_size", None) or "5 MB",
                     retention=getattr(log_conf, "backup_count", None))
        logger = get_server_logger()
        # The notices from loading the config (an old config.yaml left behind, removed
        # sections, profile warnings) were logged before the log files existed: log them again
        # so they are in logs/ too.
        Config.log_startup_notices(logger)

        # Migration-only mode: run the action and exit without starting the server.
        if args.migrate_only:
            action = args.migrate_only
            logger.info(f"🔄 Running migration-only: {action}")
            success = run_migration_command(action, args.message, args.revision)
            sys.exit(0 if success else 1)

        # DB bootstrap: ensure the DB exists and extensions are enabled (unless --no-migrate).
        if not args.no_migrate:
            db_conf = Config.get_database_config()
            if not db_conf:
                raise RuntimeError("Database config missing")
            # Resolved URL (database.port applied), same as every other connection path.
            from src.utils.db_url import resolve_db_url
            db_url = resolve_db_url(db_conf)
            try:
                from src.utils.db_bootstrap import ensure_database_ready
                ensure_database_ready(db_url, logger)
            except Exception as e:
                logger.error(f"❌ DB bootstrap failed: {e}")
                logger.error("   Check Postgres is running and DATABASE_URL (.env / config/database.yaml) is correct")
                sys.exit(1)

            # Compare the vector-table dimension against config to catch a model swap or
            # a mistyped dimension early.
            try:
                cfg_model = Config.get_config_model()
                rag_conf = getattr(cfg_model, "rag", None)
                if rag_conf and rag_conf.embedding:
                    from src.utils.db_bootstrap import check_vector_dims
                    check_vector_dims(db_url, rag_conf.embedding.dimension, logger)
            except Exception as e:
                logger.warning(f"⚠️ Vector dim startup check failed (continuing): {e}")

            # Auto-migrate to align the schema with head.
            logger.info("🔄 Running auto migration...")
            success = run_migration_command("auto")
            if success:
                logger.info("✅ Auto migration complete")
            else:
                logger.error("❌ Auto migration failed — aborting startup")
                logger.error("   Use --no-migrate to skip migration if this is intentional")
                sys.exit(1)
        else:
            logger.warning("⚠️ Skipping bootstrap + migration (--no-migrate)")

        # Overlay runtime setting overrides (admin-panel hot changes) onto the ConfigModel.
        # Must run before any RAG component is lazily constructed; skipped safely when the
        # DB is unavailable, falling back to yaml defaults.
        try:
            from src.adapter.runtime_settings_service import load_overrides_on_startup
            if args.no_migrate:
                # The tables are defined by Alembic (20261102a1b2c5); create them here
                # only when migrations were skipped, as a fallback.
                from db.runtime_settings_db import RuntimeSettingsDB
                from db.settings_audit_db import SettingsAuditDB
                from db.query_log_db import QueryLogDB
                RuntimeSettingsDB.ensure_table()
                SettingsAuditDB.ensure_table()
                QueryLogDB.ensure_table()
            load_overrides_on_startup()
        except Exception as e:
            logger.warning(f"⚠️ Runtime settings overlay skipped: {e}")

        server_conf = Config.get_server_config()
        if not server_conf:
            raise RuntimeError("Server config missing")

        host = args.host or getattr(server_conf, "host", "0.0.0.0")
        port = args.port or getattr(server_conf, "port", 5032)
        transport = args.transport or getattr(server_conf, "transport", "http")

        for change in getattr(config_obj.app, "renamed_defaults", []) if config_obj.app else []:
            logger.warning(
                f"⚠️ the config still has the pre-Folio default ({change}); "
                "update config/system.yaml or config/mcp.yaml to silence this warning")

        logger.info(f"Starting {getattr(config_obj.app, 'title', None) or 'Folio MCP Server'}...")
        logger.info(f"Config: {args.config_dir}/ (+ .env)")
        logger.info(f"URL: http://{host}:{port}")
        mcp_endpoint = "mcp" if transport == "http" else transport
        logger.info(f"MCP Endpoint: http://{host}:{port}/{mcp_endpoint}")
        logger.info(f"Transport: {transport}")

        # DB connection check; schema is handled by migrations, so no create_tables.
        try:
            engine = db_module.init_db(create_tables=False)
            with engine.connect():
                logger.info("✅ Database connection verified")
        except Exception as e:
            logger.error(f"❌ DB connect failed: {e}")
            logger.warning("⚠️ Server starting anyway — RAG features will fail")

        # Say where every model role runs (no "weights missing" noise for roles that
        # are cloud endpoints), then pre-warm docling only if a role needs its local models.
        rag_conf = getattr(config_obj, "rag", None)
        from src.config.roles import docling_warmup_needed, startup_lines
        if rag_conf is not None:
            for line in startup_lines(rag_conf):
                logger.info(line)
        if docling_warmup_needed(rag_conf):
            # Pre-warm Docling (1-2 min on first start).
            try:
                from src.domain.rag.docling_loader import warmup_docling
                warmup_docling()
            except Exception as e:
                logger.warning(f"Docling warmup skipped: {e}")
        else:
            logger.info("Docling warm-up skipped: no role parses PDFs/images or audio with local models")

        app_instance = create_app(config_obj, transport=transport)
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            log_level=log_level.lower(),
            access_log=False,  # client origin/status is already logged during token validation
        )

    except ConfigFileError as e:
        from loguru import logger as base_logger
        base_logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        from loguru import logger as base_logger
        base_logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
