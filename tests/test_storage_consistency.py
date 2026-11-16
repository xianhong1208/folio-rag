"""The startup check that the database and the storage directory belong together
(src/storage/consistency.py): each one missing on its own must stop startup with a clear message,
never start healthy without the data."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from src.storage import consistency
from src.storage.consistency import (
    StorageMismatchError,
    assess,
    check_storage_matches_database,
    database_is_new,
    uploaded_files,
)


def _upload(root: Path, rel: str = "sub_u1/docs/0f1e") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _resolver(base: Path):
    return lambda rel: base / rel


class TestUploadedFiles:
    def test_files_in_the_root_do_not_count(self, tmp_path):
        """The console session secret lives directly in storage/: a new install has it."""
        (tmp_path / ".console_session_secret").write_text("s")
        (tmp_path / "sub_u1" / "empty-folder").mkdir(parents=True)
        assert uploaded_files(tmp_path) == []

    def test_an_upload_counts(self, tmp_path):
        path = _upload(tmp_path)
        assert uploaded_files(tmp_path) == [path]

    def test_a_missing_root_has_none(self, tmp_path):
        assert uploaded_files(tmp_path / "nope") == []


class TestAssess:
    def test_new_database_next_to_old_uploads_is_refused(self, tmp_path):
        storage = tmp_path / "storage"
        _upload(storage)
        error, warning = assess(database_was_new=True, total_files=0, recent_paths=[],
                                resolve=_resolver(tmp_path), storage_root=storage)
        assert error and "database is new" in error and "folio-db" in error and warning is None

    def test_new_database_and_empty_storage_is_a_new_install(self, tmp_path):
        storage = tmp_path / "storage"
        storage.mkdir()
        (storage / ".console_session_secret").write_text("s")
        assert assess(database_was_new=True, total_files=0, recent_paths=[],
                      resolve=_resolver(tmp_path), storage_root=storage) == (None, None)

    def test_database_without_any_of_its_files_is_refused(self, tmp_path):
        storage = tmp_path / "storage"
        storage.mkdir()
        error, _ = assess(database_was_new=False, total_files=3,
                          recent_paths=["storage/sub_u1/a/1", "storage/sub_u1/a/2"],
                          resolve=_resolver(tmp_path), storage_root=storage)
        assert error and "none of the 2 most recent" in error and "folio-storage" in error

    def test_some_files_missing_is_only_a_warning(self, tmp_path):
        storage = tmp_path / "storage"
        _upload(storage, "sub_u1/a/1")
        error, warning = assess(database_was_new=False, total_files=2,
                                recent_paths=["storage/sub_u1/a/1", "storage/sub_u1/a/2"],
                                resolve=_resolver(tmp_path), storage_root=storage)
        assert error is None and warning and "1 of the 2" in warning

    def test_everything_present_passes(self, tmp_path):
        storage = tmp_path / "storage"
        _upload(storage, "sub_u1/a/1")
        assert assess(database_was_new=False, total_files=1, recent_paths=["storage/sub_u1/a/1"],
                      resolve=_resolver(tmp_path), storage_root=storage) == (None, None)

    def test_existing_database_without_files_passes_whatever_storage_holds(self, tmp_path):
        """Every file deleted through the API (or stray files): not a missing database."""
        storage = tmp_path / "storage"
        _upload(storage)
        assert assess(database_was_new=False, total_files=0, recent_paths=[],
                      resolve=_resolver(tmp_path), storage_root=storage) == (None, None)


@pytest.fixture()
def sqlite_url(tmp_path):
    return f"sqlite:///{tmp_path / 'folio.db'}"


def _make_files_table(url, paths):
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE "Files" (file_path TEXT, upload_time TIMESTAMP)'))
        for i, p in enumerate(paths):
            conn.execute(text('INSERT INTO "Files" VALUES (:p, :t)'),
                         {"p": p, "t": f"2026-11-1{i} 00:00:00"})
    engine.dispose()


class TestDatabase:
    def test_database_is_new_without_schema(self, sqlite_url):
        assert database_is_new(sqlite_url) is True

    def test_alembic_version_means_existing(self, sqlite_url):
        engine = create_engine(sqlite_url)
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE alembic_version (version_num TEXT)"))
        engine.dispose()
        assert database_is_new(sqlite_url) is False

    def test_a_pre_alembic_files_table_means_existing(self, sqlite_url):
        _make_files_table(sqlite_url, [])
        assert database_is_new(sqlite_url) is False

    def test_check_refuses_a_database_whose_files_are_gone(self, sqlite_url, tmp_path):
        _make_files_table(sqlite_url, ["storage/sub_u1/a/1"])
        with pytest.raises(StorageMismatchError, match="none of the 1 most recent"):
            check_storage_matches_database(sqlite_url, database_was_new=False,
                                           storage_root=tmp_path / "storage",
                                           resolve=_resolver(tmp_path), logger=MagicMock())

    def test_check_passes_and_logs(self, sqlite_url, tmp_path):
        _make_files_table(sqlite_url, ["storage/sub_u1/a/1"])
        _upload(tmp_path / "storage", "sub_u1/a/1")
        logger = MagicMock()
        check_storage_matches_database(sqlite_url, database_was_new=False,
                                       storage_root=tmp_path / "storage",
                                       resolve=_resolver(tmp_path), logger=logger)
        assert "Storage check passed" in logger.info.call_args[0][0]

    def test_check_samples_the_most_recent_files(self, sqlite_url, tmp_path, monkeypatch):
        """Only the newest SAMPLE_SIZE are looked at: the oldest one missing is not seen."""
        monkeypatch.setattr(consistency, "SAMPLE_SIZE", 1)
        _make_files_table(sqlite_url, ["storage/sub_u1/a/old", "storage/sub_u1/a/new"])
        _upload(tmp_path / "storage", "sub_u1/a/new")
        logger = MagicMock()
        check_storage_matches_database(sqlite_url, database_was_new=False,
                                       storage_root=tmp_path / "storage",
                                       resolve=_resolver(tmp_path), logger=logger)
        logger.warning.assert_not_called()

    def test_check_warns_on_a_partial_loss(self, sqlite_url, tmp_path):
        _make_files_table(sqlite_url, ["storage/sub_u1/a/1", "storage/sub_u1/a/2"])
        _upload(tmp_path / "storage", "sub_u1/a/2")
        logger = MagicMock()
        check_storage_matches_database(sqlite_url, database_was_new=False,
                                       storage_root=tmp_path / "storage",
                                       resolve=_resolver(tmp_path), logger=logger)
        assert "1 of the 2" in logger.warning.call_args[0][0]

    def test_check_on_a_new_database_does_not_query_it(self, tmp_path):
        """Called after migrating a new database: only the storage directory decides."""
        _upload(tmp_path / "storage")
        with pytest.raises(StorageMismatchError, match="database is new"):
            check_storage_matches_database("postgresql://unused/none", database_was_new=True,
                                           storage_root=tmp_path / "storage",
                                           resolve=_resolver(tmp_path), logger=MagicMock())


class TestMain:
    def test_main_refuses_on_a_mismatch(self, monkeypatch, tmp_path):
        import main
        from src.storage.file_storage import FileStorage
        monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path / "storage")
        _upload(tmp_path / "storage")
        logger = MagicMock()
        assert main.storage_matches_database("postgresql://unused/none", True, logger) is False
        assert "database is new" in logger.error.call_args[0][0]

    def test_main_checks_a_new_database_before_migrating_it(self):
        """A new database next to old uploads is refused before it is migrated: otherwise a restart
        (restart: unless-stopped) would find a migrated, empty database and start."""
        import main
        source = Path(main.__file__).read_text(encoding="utf-8")
        new_at = source.index("database_was_new = database_is_new(db_url)")
        new_check = source.index("if database_was_new and not storage_matches_database(db_url, True, logger):")
        migrate_at = source.index('run_migration_command("auto")', new_at)
        old_check = source.index("if not database_was_new and not storage_matches_database(db_url, False, logger):")
        assert new_at < new_check < migrate_at < old_check
        for at in (new_check, old_check):
            assert source[at:].split("\n", 2)[1].strip() == "sys.exit(1)"
