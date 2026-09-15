#!/usr/bin/env python3

"""Database migration utility built on the Alembic Python API (no CLI dependency).

Run with `python -m db.migrate [command]`; see `main()` for available commands.
"""
import sys
import re
import sqlalchemy
import traceback
from pathlib import Path
from typing import Tuple, Optional, List

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from alembic.config import Config as AlembicConfig
from alembic import command

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.log import get_server_logger
from src.config.config_manager import Config
from src.utils.runtime_paths import resolve_external_dir

logger = get_server_logger()

# Cross-instance global migration lock key (pg_advisory_lock). Any fixed
# 64-bit value works as long as every instance uses the same one; prevents
# multiple workers / rolling deployments from running auto_migrate on the same
# migration concurrently (e.g. concurrent DROP TABLE would succeed on one and
# fail on another).
_PG_MIGRATE_LOCK_KEY = 74_112_358


class MigrationChainError(RuntimeError):
    """Migration parsing / chain-integrity failure — raised so the problem fails
    loudly instead of silently leaving migrations unapplied and the schema drifting.
    """


def get_external_base_dir() -> Path:
    """Return the base directory for external files (config, versions, data).

    Uses `config/` as the marker: config and versions always live as siblings
    under the same base in every deployment mode, so wherever config is, that's
    the base. Falls back to `project_root` if nothing can be resolved.
    """
    config_dir = resolve_external_dir("config", dev_root=project_root)
    if config_dir is not None:
        return config_dir.parent
    return project_root


def get_migrate_dir() -> Path:
    """Get the migrate directory path (contains alembic logic)"""
    return Path(__file__).parent / "migrate"


def get_versions_dir() -> Path:
    """Return the versions directory (migration files); external, not packaged."""
    external_base = get_external_base_dir()
    return external_base / "versions"


def get_config_dir() -> Path:
    """Get the config directory path"""
    external_base = get_external_base_dir()
    return external_base / "config"


class DatabaseMigrator:
    """Database migration manager using the Alembic Python API.

    Works in both development and packaged mode. Alembic logic lives in
    db/migrate/ (packaged); migration files live in versions/ (external).
    """

    def __init__(self):
        self.project_root = project_root
        self.external_base = get_external_base_dir()
        self.versions_dir = get_versions_dir()

        self.db_conf = Config.get_database_config()
        self.db_url = getattr(self.db_conf, 'url')
        self._engine = create_engine(self.db_url)

        logger.debug(f"External base directory: {self.external_base}")
        logger.debug(f"Versions directory: {self.versions_dir}")

        if not self.versions_dir.exists():
            logger.warning(f"Versions directory not found, creating: {self.versions_dir}")
            self.versions_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_current_revision(self) -> Optional[str]:
        """Get current database revision"""
        try:
            with self._engine.connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    @staticmethod
    def _discover_revisions(versions_dir: Path) -> List[Tuple[str, Optional[str], Path]]:
        """Parse versions/*.py into an integrity-checked chain ordered base -> head.

        Raises MigrationChainError (never silently skips) on any of: an id the
        regex cannot parse, multiple bases or heads (a fork), or an orphan whose
        down_revision points nowhere. Returns [] for an empty directory.
        """
        version_files = [
            f for f in sorted(versions_dir.glob("*.py"))
            if not f.name.startswith('__')
        ]
        if not version_files:
            return []

        revisions = {}  # revision -> (down_revision, file_path)
        unparseable: List[str] = []
        for vf in version_files:
            content = vf.read_text(encoding='utf-8')
            # The (?<!down_) lookbehind stops the match sliding onto the
            # down_revision line and reading the down value as the revision.
            rev_match = re.search(r"(?<!down_)revision(?:\s*:\s*str)?\s*=\s*['\"]([a-f0-9]+)['\"]", content)
            down_match = re.search(r"down_revision(?:\s*:[^=]+)?\s*=\s*(?:['\"]?([a-f0-9]+)['\"]?|None)", content)
            if not rev_match:
                unparseable.append(vf.name)
                continue
            rev = rev_match.group(1)
            down_rev = down_match.group(1) if down_match and down_match.group(1) else None
            revisions[rev] = (down_rev, vf)
            logger.debug(f"Found migration: {vf.name} -> rev={rev}, down={down_rev}")

        if unparseable:
            raise MigrationChainError(
                f"Cannot parse revision from migration file(s): {unparseable} — "
                f"a revision id may only contain [a-f0-9] (regex limitation), "
                f"otherwise it is silently skipped. Use a pure-hex id (e.g. 20260819cafe01)."
            )

        all_revs = set(revisions.keys())
        down_revs = set(r[0] for r in revisions.values() if r[0])

        bases = [r for r, (d, _) in revisions.items() if d is None]
        if len(bases) != 1:
            raise MigrationChainError(
                f"The migration chain must have exactly one base (down_revision=None), "
                f"got {len(bases)}: {sorted(bases)} — multiple bases mean a broken chain."
            )
        heads = sorted(all_revs - down_revs)
        if len(heads) != 1:
            raise MigrationChainError(
                f"Multiple heads detected: {heads} — the migration history has forked; "
                f"only one chain would be applied and the rest silently dropped. "
                f"Rebase the branches into a linear history (adjust one branch's down_revision)."
            )

        chain: List[Tuple[str, Optional[str], Path]] = []
        current: Optional[str] = bases[0]
        while current:
            down_rev, file_path = revisions[current]
            chain.append((current, down_rev, file_path))
            nxt = [r for r, (d, _) in revisions.items() if d == current]
            current = nxt[0] if nxt else None

        if len(chain) != len(revisions):
            in_chain = {r for r, _, _ in chain}
            orphans = sorted(
                f"{revisions[r][1].name}(down={revisions[r][0]})"
                for r in all_revs - in_chain
            )
            raise MigrationChainError(
                f"Broken chain: {len(revisions) - len(chain)} migration(s) are orphans "
                f"(down_revision points to a nonexistent rev): {orphans}"
            )
        return chain

    def _get_head_revision(self) -> Optional[str]:
        """Get head revision from migration files (raises on chain-validation failure)"""
        chain = self._discover_revisions(self.versions_dir)
        return chain[-1][0] if chain else None
    
    def _get_migration_chain(self) -> List[Tuple[str, Optional[str], Path]]:
        """Return migrations ordered base -> head as (revision, down_revision, file_path)."""
        return self._discover_revisions(self.versions_dir)

    def upgrade(self, revision: str = "head") -> bool:
        """Apply migrations to target revision
        
        Args:
            revision: Target revision (default: "head")
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"🔄 Upgrading database to: {revision}")
            
            current_rev = self._get_current_revision()
            head_rev = self._get_head_revision()
            
            logger.info(f"📍 Current: {current_rev or 'none'}")
            logger.info(f"🎯 Head: {head_rev or 'none'}")
            
            target_rev = head_rev if revision == "head" else revision
            
            if not target_rev:
                logger.info("✅ No migrations to apply")
                return True
            
            if current_rev == target_rev:
                logger.info("✅ Database is already at target revision")
                return True

            # Multi-instance guard: a session-scoped pg_advisory_lock so that when
            # multiple workers / rolling deployments start at once, only one runs
            # the migration while the rest wait here.
            with self._engine.connect() as lock_conn:
                lock_conn.execute(
                    text("SELECT pg_advisory_lock(:k)"), {"k": _PG_MIGRATE_LOCK_KEY}
                )
                try:
                    # Re-check after acquiring the lock: a competitor may already
                    # have upgraded to the target while we waited.
                    current_rev = self._get_current_revision()
                    if current_rev == target_rev:
                        logger.info("✅ Another instance already upgraded to target — skipping")
                        return True

                    chain = self._get_migration_chain()
                    if not chain:
                        logger.warning("⚠️ No migration files found")
                        return True

                    start_applying = current_rev is None
                    migrations_to_apply = []

                    for rev, down_rev, file_path in chain:
                        if start_applying:
                            migrations_to_apply.append((rev, file_path))
                        elif down_rev == current_rev:
                            start_applying = True
                            migrations_to_apply.append((rev, file_path))

                        if rev == target_rev:
                            break

                    if not migrations_to_apply:
                        logger.info("✅ No migrations to apply")
                        return True

                    logger.info(f"📋 Applying {len(migrations_to_apply)} migration(s)...")

                    for rev, file_path in migrations_to_apply:
                        logger.info(f"  ⏳ Applying: {rev} ({file_path.name})")

                        if not self._execute_upgrade(self._engine, rev, file_path):
                            logger.error(f"  ❌ Failed to apply: {rev}")
                            return False

                        logger.info(f"  ✅ Applied: {rev}")

                    logger.info(f"✅ Successfully upgraded to: {target_rev}")
                    return True
                finally:
                    lock_conn.execute(
                        text("SELECT pg_advisory_unlock(:k)"), {"k": _PG_MIGRATE_LOCK_KEY}
                    )
            
        except Exception as e:
            logger.error(f"❌ Upgrade failed: {e}")
            traceback.print_exc()
            return False
    
    def _execute_upgrade(self, engine: Engine, revision: str, file_path: Path) -> bool:
        """Execute a single migration's upgrade function"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"migration_{revision}", file_path)
            if not spec or not spec.loader:
                logger.error(f"Cannot load migration module: {file_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)

            from alembic import op
            module.op = op
            module.sa = sqlalchemy

            spec.loader.exec_module(module)

            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS alembic_version (
                        version_num VARCHAR(32) NOT NULL,
                        CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
                    )
                """))

                context = MigrationContext.configure(conn)

                with Operations.context(context):
                    if hasattr(module, 'upgrade'):
                        module.upgrade()
                    else:
                        logger.warning(f"No upgrade function in {file_path}")

                conn.execute(text("DELETE FROM alembic_version"))
                conn.execute(
                    text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                    {"rev": revision}
                )

            return True

        except Exception as e:
            logger.error(f"Failed to execute migration {revision}: {e}")
            traceback.print_exc()
            return False

    def downgrade(self, revision: str = "-1") -> bool:
        """Rollback migrations to target revision
        
        Args:
            revision: Target revision ("-1" for previous, "base" for all)
            
        Returns:
            True if successful
        """
        try:
            logger.warning(f"⚠️ Downgrading database to: {revision}")
            
            current_rev = self._get_current_revision()
            
            if not current_rev:
                logger.info("✅ Database has no revision to downgrade from")
                return True
            
            # Resolve a relative revision (-1, -2, ...) to an absolute one
            if revision.startswith("-"):
                steps = int(revision)
                chain = self._get_migration_chain()
                
                current_idx = None
                for i, (rev, _, _) in enumerate(chain):
                    if rev == current_rev:
                        current_idx = i
                        break
                
                if current_idx is None:
                    logger.error(f"Current revision {current_rev} not found in migration chain")
                    return False
                
                target_idx = current_idx + steps
                if target_idx < 0:
                    revision = "base"
                else:
                    revision = chain[target_idx][0]
            
            logger.info(f"📍 Current: {current_rev}")
            logger.info(f"🎯 Target: {revision}")

            chain = self._get_migration_chain()
            migrations_to_downgrade = []
            found_current = False

            for rev, down_rev, file_path in reversed(chain):
                if rev == current_rev:
                    found_current = True

                if found_current:
                    migrations_to_downgrade.append((rev, down_rev, file_path))
                    if revision == "base" and down_rev is None:
                        break
                    if down_rev == revision:
                        break

            if not migrations_to_downgrade:
                logger.info("✅ No migrations to downgrade")
                return True

            logger.info(f"📋 Downgrading {len(migrations_to_downgrade)} migration(s)...")

            for rev, down_rev, file_path in migrations_to_downgrade:
                logger.info(f"  ⏳ Downgrading: {rev} ({file_path.name})")

                if not self._execute_downgrade(self._engine, rev, down_rev, file_path):
                    logger.error(f"  ❌ Failed to downgrade: {rev}")
                    return False

                logger.info(f"  ✅ Downgraded: {rev}")

            logger.info(f"✅ Successfully downgraded to: {revision}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Downgrade failed: {e}")
            traceback.print_exc()
            return False
    
    def _execute_downgrade(self, engine: Engine, revision: str, down_revision: Optional[str], file_path: Path) -> bool:
        """Execute a single migration's downgrade function"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"migration_{revision}", file_path)
            if not spec or not spec.loader:
                logger.error(f"Cannot load migration module: {file_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            
            from alembic import op
            module.op = op
            module.sa = sqlalchemy
            
            spec.loader.exec_module(module)
            
            with engine.begin() as conn:
                context = MigrationContext.configure(conn)
                
                with Operations.context(context):
                    if hasattr(module, 'downgrade'):
                        module.downgrade()
                    else:
                        logger.warning(f"No downgrade function in {file_path}")

                conn.execute(text("DELETE FROM alembic_version"))
                if down_revision:
                    conn.execute(
                        text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                        {"rev": down_revision}
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to execute downgrade {revision}: {e}")
            traceback.print_exc()
            return False

    def stamp(self, revision: str = "head") -> bool:
        """Stamp database with revision without running migrations
        
        Args:
            revision: Revision to stamp (default: "head")
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"🔖 Stamping database at: {revision}")
            
            if revision == "head":
                revision = self._get_head_revision()
                if not revision:
                    logger.warning("No migrations found, nothing to stamp")
                    return True
            
            with self._engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS alembic_version (
                        version_num VARCHAR(32) NOT NULL,
                        PRIMARY KEY (version_num)
                    )
                """))

                conn.execute(text("DELETE FROM alembic_version"))
                conn.execute(
                    text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                    {"rev": revision}
                )

            logger.info(f"✅ Successfully stamped at: {revision}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Stamp failed: {e}")
            traceback.print_exc()
            return False

    def check_status(self) -> Tuple[bool, str, str]:
        """Check migration status
        
        Returns:
            Tuple of (is_up_to_date, current_revision, head_revision)
        """
        logger.info("📊 Checking migration status...")
        
        try:
            current_rev = self._get_current_revision() or "none"
            head_rev = self._get_head_revision() or "none"
            
            is_up_to_date = (current_rev == head_rev and current_rev != "none")
            
            print("=" * 50)
            print("DATABASE MIGRATION STATUS")
            print("=" * 50)
            print(f"Current revision: {current_rev}")
            if is_up_to_date:
                print("                 (head)")
            print(f"Latest revision:  {head_rev}")
            print()
            
            if is_up_to_date:
                logger.info("✅ Database is up to date")
            elif current_rev == "none":
                logger.warning("⚠️ Database not initialized")
                print("💡 Run migrations to initialize")
            else:
                logger.warning("⚠️ Database needs migration")
                print("💡 Run migrations to update")
            
            return is_up_to_date, current_rev, head_rev
            
        except Exception as e:
            logger.error(f"❌ Error checking status: {e}")
            return False, "error", "error"

    def auto_migrate(self) -> bool:
        """Automatically apply pending migrations
        
        This is the main entry point for automatic migration handling.
        
        Returns:
            True if successful
        """
        logger.info("🚀 Starting automatic migration process...")
        
        try:
            current_rev = self._get_current_revision()
            head_rev = self._get_head_revision()
            
            logger.info(f"📍 Current revision: {current_rev or 'none'}")
            logger.info(f"🎯 Head revision: {head_rev or 'none'}")
            
            if current_rev == head_rev:
                if current_rev:
                    logger.info("✅ Database is up to date")
                else:
                    logger.warning("⚠️ No migrations found")
                return True
            
            # Apply migrations
            logger.info("⬆️ Applying migrations...")
            if self.upgrade("head"):
                logger.info("✅ Migrations applied successfully")
                return True
            else:
                logger.error("❌ Migration failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Auto migration failed: {e}")
            traceback.print_exc()
            return False

    def show_current_revision(self) -> bool:
        """Show current database revision"""
        logger.info("📍 Current database revision:")
        current = self._get_current_revision()
        print(current if current else "(no revision - database not initialized)")
        return True

    def show_heads(self) -> bool:
        """Show head revisions"""
        logger.info("🎯 Head revisions:")
        head = self._get_head_revision()
        print(head if head else "(no head revision)")
        return True

    # Alias methods for main.py compatibility

    def apply_migrations(self) -> bool:
        """Apply all pending migrations (alias for upgrade('head'))"""
        return self.upgrade("head")
    
    def rollback_migration(self, revision: str = "-1") -> bool:
        """Rollback migration (alias for downgrade)"""
        return self.downgrade(revision)
    
    def initialize_database(self) -> bool:
        """Initialize database - create tables and stamp to head
        
        Used for fresh database setup.
        """
        try:
            logger.info("🔧 Initializing database...")

            from db.db import Base

            logger.info("Creating database tables...")
            Base.metadata.create_all(self._engine)

            logger.info("Stamping database to head revision...")
            self.stamp("head")
            
            logger.info("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    def create_migration(self, message: str) -> bool:
        """Create a new Alembic migration file (autogenerate).

        Only available in dev environments (requires the Alembic CLI and an
        existing migrate_dir). Packaged environments should generate migrations
        before packaging.

        Args:
            message: Revision message (becomes part of the filename).

        Returns:
            True on success; False if migrate_dir is missing or Alembic fails.
        """
        try:
            logger.info(f"📝 Creating migration: {message}")

            migrate_dir = get_migrate_dir()
            if not migrate_dir.exists():
                logger.error("❌ Migrate directory not found. Cannot create migrations in packaged mode.")
                return False

            alembic_ini_path = migrate_dir / "alembic.ini"
            if alembic_ini_path.exists():
                alembic_cfg = AlembicConfig(str(alembic_ini_path))
            else:
                alembic_cfg = AlembicConfig()
                alembic_cfg.set_main_option("script_location", str(migrate_dir / "alembic"))

            # Alembic must use the external versions dir and the configured DB URL
            alembic_cfg.set_main_option("version_locations", str(self.versions_dir))
            alembic_cfg.set_main_option("sqlalchemy.url", str(self.db_url))

            command.revision(alembic_cfg, message=message, autogenerate=True)

            logger.info("✅ Migration created successfully (via Alembic API)")
            return True

        except ImportError:
            logger.error("❌ Alembic package not installed. Install with: pip install alembic")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create migration: {e}")
            traceback.print_exc()
            return False
    
    def show_migration_history(self) -> bool:
        """Show migration history"""
        try:
            logger.info("📜 Migration history:")
            
            chain = self._get_migration_chain()
            current = self._get_current_revision()
            
            if not chain:
                print("(no migrations found)")
                return True
            
            for rev, down_rev, file_path in chain:
                marker = " <- current" if rev == current else ""
                down_str = down_rev if down_rev else "(base)"
                print(f"  {down_str} -> {rev}  [{file_path.name}]{marker}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to show migration history: {e}")
            return False


def main():
    """Command line interface for database migration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Database Migration Tool (Pure Python API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m db.migrate                  # Auto migration (default)
    python -m db.migrate migrate          # Apply pending migrations
    python -m db.migrate status           # Show status
    python -m db.migrate rollback -1      # Rollback one migration
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("auto", help="Auto-apply migrations")
    subparsers.add_parser("migrate", aliases=["upgrade"], help="Apply pending migrations")
    
    rollback_parser = subparsers.add_parser("rollback", aliases=["downgrade"], help="Rollback migration")
    rollback_parser.add_argument("--revision", "-r", default="-1", help="Revision to rollback to")
    
    subparsers.add_parser("status", help="Show migration status")
    subparsers.add_parser("current", help="Show current revision")
    subparsers.add_parser("heads", help="Show head revisions")
    subparsers.add_parser("init", aliases=["stamp"], help="Initialize database to current state")
    
    args = parser.parse_args()
    
    if not args.command:
        args.command = "auto"
    
    try:
        migrator = DatabaseMigrator()
        success = False
        
        if args.command == "auto":
            success = migrator.auto_migrate()
        elif args.command in ["migrate", "upgrade"]:
            success = migrator.upgrade("head")
        elif args.command in ["rollback", "downgrade"]:
            success = migrator.downgrade(args.revision)
        elif args.command == "status":
            migrator.check_status()
            success = True
        elif args.command == "current":
            success = migrator.show_current_revision()
        elif args.command == "heads":
            success = migrator.show_heads()
        elif args.command in ["init", "stamp"]:
            success = migrator.stamp("head")
        else:
            parser.print_help()
            success = False
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
