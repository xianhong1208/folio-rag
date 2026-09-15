
"""Unit tests for src/storage/file_storage.py

No dependency on the DB / vLLM / any external service — all filesystem
operations are directed at pytest's tmp_path. Note: importing the module runs
_resolve_storage_root() to decide STORAGE_ROOT, so tests monkeypatch
FileStorage's class attribute directly, while _resolve_storage_root() itself is
tested separately via an env var.
"""

from pathlib import Path

import pytest

import src.storage.file_storage as fs_mod
from src.storage.file_storage import FileStorage, _resolve_storage_root, _validate_safe_name
from src.utils.runtime_paths import resolve_base_dir


@pytest.fixture
def storage_root(tmp_path, monkeypatch) -> Path:
    """Direct FileStorage's storage root to tmp_path so tests do not touch the real storage/."""
    monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path / "storage")
    monkeypatch.setattr(FileStorage, "STORAGE_FILE_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "name",
    [
        "abc",
        "ABC-123_x",
        "report.pdf",
        "中文資料夾",
        "財報 2026.xlsx",
        "mixed中英-1.0",
    ],
)
def test_validate_safe_name_accepts_legal_names(name):
    """Legal names (alphanumeric / _-. / space / CJK) pass without raising (TC-file-storage-01)."""
    _validate_safe_name(name, "folder")  # should not raise


def test_validate_safe_name_rejects_empty_string():
    """An empty string raises ValueError (TC-file-storage-02)."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_safe_name("", "token")


@pytest.mark.parametrize("bad", [None, 123, 1.5, ["a"]])
def test_validate_safe_name_rejects_non_string(bad):
    """None / a non-string type raises ValueError (TC-file-storage-03)."""
    with pytest.raises(ValueError):
        _validate_safe_name(bad, "token")


@pytest.mark.parametrize("bad", ["..", "../etc/passwd", "a..b", "a/../b"])
def test_validate_safe_name_rejects_dotdot_traversal(bad):
    """A path-traversal name containing `..` raises ValueError (TC-file-storage-04)."""
    with pytest.raises(ValueError, match="illegal path characters"):
        _validate_safe_name(bad, "folder")


@pytest.mark.parametrize("bad", ["a/b", "/abs", "a\\b", "\\\\share"])
def test_validate_safe_name_rejects_path_separators(bad):
    """A name containing a `/` or `\\` separator raises ValueError (TC-file-storage-05)."""
    with pytest.raises(ValueError, match="illegal path characters"):
        _validate_safe_name(bad, "folder")


@pytest.mark.parametrize("bad", [".hidden", ".env"])
def test_validate_safe_name_rejects_leading_dot(bad):
    """A name starting with `.` raises ValueError (TC-file-storage-06)."""
    with pytest.raises(ValueError, match="illegal path characters"):
        _validate_safe_name(bad, "file")


@pytest.mark.parametrize("bad", ["a*b", "a?b", "a$b", "a:b", "a|b", "🚀rocket", "a\tb"])
def test_validate_safe_name_rejects_special_characters(bad):
    """Special characters / emoji outside the whitelist raise ValueError (TC-file-storage-07)."""
    with pytest.raises(ValueError, match="illegal characters"):
        _validate_safe_name(bad, "file")


def test_resolve_storage_root_env_override(tmp_path, monkeypatch):
    """When FOLIO_STORAGE_ROOT is set, that path is used directly (TC-file-storage-08)."""
    custom = tmp_path / "custom_root"
    monkeypatch.setenv("FOLIO_STORAGE_ROOT", str(custom))
    assert _resolve_storage_root() == custom.resolve()


def test_resolve_storage_root_defaults_to_base_dir(monkeypatch):
    """Without the env var, it returns resolve_base_dir (dev mode = project root) (TC-file-storage-09)."""
    monkeypatch.delenv("FOLIO_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("NUITKA_ONEFILE_PARENT", raising=False)
    project_root = Path(fs_mod.__file__).resolve().parents[2]
    assert _resolve_storage_root() == resolve_base_dir(dev_root=project_root)


def test_save_file_writes_content_and_returns_relative_path(storage_root):
    """save_file writes the file and returns a relative path starting with storage/ (TC-file-storage-10)."""
    rel = FileStorage.save_file("tok1", "月報", "file-uuid-1", b"hello bytes")
    assert rel == "storage/tok1/月報/file-uuid-1"
    saved = storage_root / "storage" / "tok1" / "月報" / "file-uuid-1"
    assert saved.read_bytes() == b"hello bytes"


def test_save_file_illegal_folder_name_raises_ioerror(storage_root):
    """When save_file gets an illegal folder_name, the ValueError is wrapped as IOError (TC-file-storage-11)."""
    with pytest.raises(IOError, match="Failed to save file"):
        FileStorage.save_file("tok1", "../escape", "fid", b"x")
    # Nothing should be created outside the tmp root
    assert not (storage_root / "escape").exists()


def test_read_file_returns_saved_content(storage_root):
    """read_file reads back the original content using the relative path returned by save_file (TC-file-storage-12)."""
    rel = FileStorage.save_file("tok1", "docs", "fid-2", b"\x00\x01binary")
    assert FileStorage.read_file(rel) == b"\x00\x01binary"


def test_read_file_missing_raises_file_not_found(storage_root):
    """read_file on a nonexistent path raises FileNotFoundError (TC-file-storage-13)."""
    with pytest.raises(FileNotFoundError):
        FileStorage.read_file("storage/tok1/docs/ghost")


def test_delete_file_success_returns_true(storage_root):
    """delete_file on an existing file returns True and the file is gone (TC-file-storage-14)."""
    rel = FileStorage.save_file("tok1", "docs", "fid-3", b"x")
    assert FileStorage.delete_file(rel) is True
    assert not (storage_root / rel).exists()


def test_delete_file_missing_returns_false(storage_root):
    """delete_file on a nonexistent file returns False without raising (TC-file-storage-15)."""
    assert FileStorage.delete_file("storage/tok1/docs/ghost") is False


def test_delete_folder_removes_recursively(storage_root):
    """delete_folder recursively removes the entire folder and returns True (TC-file-storage-16)."""
    FileStorage.save_file("tok1", "docs", "fid-a", b"1")
    FileStorage.save_file("tok1", "docs", "fid-b", b"2")
    assert FileStorage.delete_folder("tok1", "docs") is True
    assert not (storage_root / "storage" / "tok1" / "docs").exists()


def test_delete_folder_missing_returns_false(storage_root):
    """delete_folder on a nonexistent folder returns False (TC-file-storage-17)."""
    assert FileStorage.delete_folder("tok1", "ghost") is False


def test_delete_folder_illegal_name_raises_value_error(storage_root):
    """delete_folder raises ValueError on an illegal name (not swallowed into False) (TC-file-storage-18)."""
    with pytest.raises(ValueError):
        FileStorage.delete_folder("tok1", "../../etc")


def test_rename_folder_success(storage_root):
    """A successful rename_folder returns True; the old path is gone and the new path keeps the files (TC-file-storage-19)."""
    FileStorage.save_file("tok1", "old_name", "fid", b"data")
    assert FileStorage.rename_folder("tok1", "old_name", "new_name") is True
    base = storage_root / "storage" / "tok1"
    assert not (base / "old_name").exists()
    assert (base / "new_name" / "fid").read_bytes() == b"data"


def test_rename_folder_source_missing_returns_false(storage_root):
    """rename_folder returns False when the source folder does not exist (TC-file-storage-20)."""
    assert FileStorage.rename_folder("tok1", "ghost", "new") is False


def test_rename_folder_target_exists_returns_false(storage_root):
    """rename_folder returns False and does not overwrite when the target folder already exists (TC-file-storage-21)."""
    FileStorage.save_file("tok1", "src_dir", "fid1", b"1")
    FileStorage.save_file("tok1", "dst_dir", "fid2", b"2")
    assert FileStorage.rename_folder("tok1", "src_dir", "dst_dir") is False
    base = storage_root / "storage" / "tok1"
    assert (base / "src_dir" / "fid1").exists()
    assert (base / "dst_dir" / "fid2").exists()


def test_resolve_path_joins_relative_under_file_root(storage_root):
    """resolve_path joins the DB relative path under STORAGE_FILE_ROOT (TC-file-storage-22)."""
    assert FileStorage.resolve_path("storage/t/f/id") == storage_root / "storage" / "t" / "f" / "id"


def test_resolve_path_absolute_input_kept_as_is(storage_root):
    """With an absolute input, Path semantics discard the left side and return it as-is (TC-file-storage-23)."""
    assert FileStorage.resolve_path("/abs/x/y") == Path("/abs/x/y")


def test_read_file_not_found_raises():
    """Reading a nonexistent file: log and re-raise FileNotFoundError unchanged (not swallowed or converted)."""
    from src.storage.file_storage import FileStorage
    import pytest as _pytest
    with _pytest.raises(FileNotFoundError):
        FileStorage.read_file("storage/no-such-token/no-such-file-xyz")
