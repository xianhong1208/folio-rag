"""File storage management on the local filesystem.

Storage layout: storage/{user_token}/{folder_name}/{file_id}
"""

import os
import re
import shutil
from pathlib import Path
from src.log import get_api_logger
from src.utils.runtime_paths import resolve_base_dir

logger = get_api_logger()


def _resolve_storage_root() -> Path:
    """Resolve the storage root directory.

    Order: the FOLIO_STORAGE_ROOT env var (explicit override), else
    resolve_base_dir() (the same onefile-aware logic used for config/assets).
    resolve_base_dir is preferred over cwd / __file__ because under Nuitka
    --onefile __file__ points at an ephemeral /tmp path, and cwd is unpredictable
    under systemd. Every deployment works without setting the env var.
    """
    env_root = os.getenv("FOLIO_STORAGE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return resolve_base_dir(dev_root=Path(__file__).resolve().parents[2])


_PROJECT_ROOT = _resolve_storage_root()
logger.info(f"[INIT] Storage root resolved: {_PROJECT_ROOT}")

# Safe characters for folder/token names: letters / digits / dash / underscore /
# dot / space / CJK. Guards against path traversal such as ../etc/passwd.
_SAFE_NAME_PATTERN = re.compile(r'^[\w\-. 一-鿿]+$')


def _validate_safe_name(name: str, kind: str = "name") -> None:
    """Validate a name to prevent path traversal.

    Must be non-empty, contain only allowed characters, not start with ``.``,
    and not contain ``..``. Raises ValueError on violation.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"{kind} cannot be empty")
    if '..' in name or name.startswith('.') or '/' in name or '\\' in name:
        raise ValueError(f"{kind} contains illegal path characters: {name!r}")
    if not _SAFE_NAME_PATTERN.match(name):
        raise ValueError(
            f"{kind} contains illegal characters (allowed: letters, digits, _-. and CJK): {name!r}"
        )


class FileStorage:
    """File storage management class."""

    # Fixed at storage/ under the project root, not Path.cwd(): a shifting
    # startup cwd would otherwise relocate storage.
    STORAGE_ROOT = _PROJECT_ROOT / "storage"
    STORAGE_FILE_ROOT = _PROJECT_ROOT
    
    @classmethod
    def _ensure_storage_root(cls):
        """Ensure the storage root directory exists."""
        cls.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def resolve_path(cls, file_path: str) -> Path:
        """Resolve a DB-stored relative path (``storage/...``) to an absolute path.

        Keeps direct-``open()`` consumers (e.g. HierarchicalIndexer) consistent
        with save_file regardless of cwd. An already-absolute input is returned
        as-is (Path discards the left side), which is safe.
        """
        return cls.STORAGE_FILE_ROOT / file_path
    
    @classmethod
    def _get_folder_storage_path(cls, user_token: str, folder_name: str) -> Path:
        """Return the storage path for a folder, creating it.

        Raises ValueError if user_token or folder_name fails safe-name validation.
        """
        _validate_safe_name(user_token, "user_token")
        _validate_safe_name(folder_name, "folder_name")
        cls._ensure_storage_root()
        folder_path = cls.STORAGE_ROOT / user_token / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    @classmethod
    def save_file(cls, user_token: str, folder_name: str, file_id: str, file_content: bytes) -> str:
        """Save a file and return its path relative to the project root.

        Raises IOError if saving fails.
        """
        try:
            folder_path = cls._get_folder_storage_path(user_token, folder_name)
            file_path = folder_path / file_id

            with open(file_path, 'wb') as f:
                f.write(file_content)

            relative_path = f"storage/{user_token}/{folder_name}/{file_id}"
            logger.info(f"File saved successfully: {relative_path}")
            return relative_path

        except Exception as e:
            logger.error(f"Failed to save file {file_id} to {folder_name}: {str(e)}")
            raise IOError(f"Failed to save file: {str(e)}")
    
    @classmethod
    def read_file(cls, file_path: str) -> bytes:
        """Read a file (path relative to the project root).

        Raises FileNotFoundError if missing, IOError on other failures.
        """
        try:
            full_path = cls.STORAGE_FILE_ROOT / file_path

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(full_path, 'rb') as f:
                content = f.read()
            
            logger.info(f"File read successfully: {file_path}")
            return content
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            raise IOError(f"Failed to read file: {str(e)}")
    
    @classmethod
    def delete_file(cls, file_path: str) -> bool:
        """Delete a file (path relative to the project root); return whether it succeeded."""
        try:
            full_path = cls.STORAGE_FILE_ROOT / file_path

            if not full_path.exists():
                logger.warning(f"File not found for deletion: {file_path}")
                return False

            full_path.unlink()
            logger.info(f"File deleted successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False
    
    @classmethod
    def delete_folder(cls, user_token: str, folder_name: str) -> bool:
        """Delete an entire folder; return whether it succeeded."""
        try:
            _validate_safe_name(user_token, "user_token")
            _validate_safe_name(folder_name, "folder_name")
            folder_path = cls.STORAGE_ROOT / user_token / folder_name
            if not folder_path.exists():
                logger.warning(f"Folder not found for deletion: {user_token[:8] if user_token else 'N/A'}.../{folder_name}")
                return False

            shutil.rmtree(folder_path)
            logger.info(f"Folder deleted successfully: {user_token[:8] if user_token else 'N/A'}.../{folder_name}")
            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete folder {user_token[:8] if user_token else 'N/A'}.../{folder_name}: {str(e)}")
            return False

    @classmethod
    def rename_folder(cls, user_token: str, old_folder_name: str, new_folder_name: str) -> bool:
        """Rename a folder; return whether it succeeded."""
        try:
            _validate_safe_name(user_token, "user_token")
            _validate_safe_name(old_folder_name, "old_folder_name")
            _validate_safe_name(new_folder_name, "new_folder_name")
            old_path = cls.STORAGE_ROOT / user_token / old_folder_name
            new_path = cls.STORAGE_ROOT / user_token / new_folder_name

            if not old_path.exists():
                logger.warning(f"Folder not found for renaming: {user_token[:8] if user_token else 'N/A'}.../{old_folder_name}")
                return False

            if new_path.exists():
                logger.error(f"Target folder already exists: {user_token[:8] if user_token else 'N/A'}.../{new_folder_name}")
                return False

            old_path.rename(new_path)
            logger.info(f"Folder renamed successfully: {user_token[:8] if user_token else 'N/A'}.../{old_folder_name} -> {new_folder_name}")
            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to rename folder {user_token[:8] if user_token else 'N/A'}.../{old_folder_name} to {new_folder_name}: {str(e)}")
            return False
