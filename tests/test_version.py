
"""src/version.py — the three version-resolution paths (metadata → pyproject fallback → 0.0.0)."""

from unittest.mock import patch

import pytest

import src.version as v


@pytest.fixture(autouse=True)
def _clear_cache():
    v.get_version.cache_clear()
    yield
    v.get_version.cache_clear()


def test_metadata_path():
    """(1) importlib.metadata available → return directly (the primary path)."""
    with patch.object(v, "_pkg_version", return_value="9.9.9") as m:
        assert v.get_version() == "9.9.9"
    m.assert_called_once_with("agentic-rag")


def test_pyproject_fallback(tmp_path):
    """(2) metadata missing → read [project].version from the project-root pyproject.toml."""
    py = tmp_path / "pyproject.toml"
    py.write_text('[project]\nname = "x"\nversion = "7.7.7"\n', encoding="utf-8")
    with patch.object(v, "_pkg_version", side_effect=v.PackageNotFoundError), \
         patch.object(v, "_PYPROJECT_PATH", py):
        assert v.get_version() == "7.7.7"


def test_last_resort_zero(tmp_path):
    """(3) both fail → '0.0.0' plus a warning; the service can still start."""
    with patch.object(v, "_pkg_version", side_effect=v.PackageNotFoundError), \
         patch.object(v, "_PYPROJECT_PATH", tmp_path / "nonexistent.toml"):
        assert v.get_version() == "0.0.0"


def test_pyproject_malformed_falls_to_zero(tmp_path):
    """pyproject exists but lacks the version field (the KeyError path) → '0.0.0'."""
    py = tmp_path / "pyproject.toml"
    py.write_text('[project]\nname = "x"\n', encoding="utf-8")
    with patch.object(v, "_pkg_version", side_effect=v.PackageNotFoundError), \
         patch.object(v, "_PYPROJECT_PATH", py):
        assert v.get_version() == "0.0.0"


def test_result_is_cached():
    """lru_cache: the second call no longer hits metadata."""
    with patch.object(v, "_pkg_version", return_value="1.2.3") as m:
        assert v.get_version() == "1.2.3"
        assert v.get_version() == "1.2.3"
    assert m.call_count == 1
