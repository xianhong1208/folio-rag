"""File storage management on the local filesystem.

Storage layout: storage/{owner_dir}/{folder_name}/{file_id}, where ``owner_dir`` is
derived from the folder's owner key by :func:`owner_dir_name` (``sub:<sub>`` ->
``sub_<sub>``; a legacy per-token key is used as-is).
"""

import hashlib
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


_OWNER_DIR_SUB_MAX = 200


def owner_dir_name(owner_key: str) -> str:
    """Storage directory name for a folder owner key (path-safe, deterministic, injective).

    - ``sub:<sub>`` (a person) -> ``sub_<sub>`` when the sub is a safe name of at most
      200 characters (MCP Gate's user ids are UUIDs), else ``subh_<sha256 of the sub>``.
      ``:`` is not a safe path character, hence the mapping; the two prefixes cannot
      produce the same directory.
    - anything else (a legacy per-token jti, ``console`` in open mode) is used as-is
      after the safe-name check, so existing folders keep their directories.

    Raises ValueError when a legacy key is not a safe name.
    """
    from src.auth.owner_key import OWNER_PREFIX
    if owner_key and owner_key.startswith(OWNER_PREFIX):
        sub = owner_key[len(OWNER_PREFIX):]
        try:
            _validate_safe_name(sub, "sub")
            if len(sub) <= _OWNER_DIR_SUB_MAX:
                return f"sub_{sub}"
        except ValueError:
            pass
        return "subh_" + hashlib.sha256(sub.encode("utf-8")).hexdigest()
    _validate_safe_name(owner_key, "user_token")
    return owner_key


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
        owner_dir = owner_dir_name(user_token)
        _validate_safe_name(folder_name, "folder_name")
        cls._ensure_storage_root()
        folder_path = cls.STORAGE_ROOT / owner_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    @classmethod
    def folder_dir(cls, user_token: str, folder_name: str) -> Path:
        """Absolute storage directory of a folder (not created)."""
        _validate_safe_name(folder_name, "folder_name")
        return cls.STORAGE_ROOT / owner_dir_name(user_token) / folder_name

    @staticmethod
    def relative_folder_path(user_token: str, folder_name: str) -> str:
        """Prefix of a folder's stored file paths: ``storage/<owner_dir>/<folder>/``."""
        return f"storage/{owner_dir_name(user_token)}/{folder_name}/"

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

            relative_path = cls.relative_folder_path(user_token, folder_name) + str(file_id)
            logger.info(f"File saved successfully: {relative_path}")
            return relative_path

        except Exception as e:
            logger.error(f"Failed to save file {file_id} to {folder_name}: {str(e)}")
            raise IOError(f"Failed to save file: {str(e)}")
    
    # Copy uploads in 1 MiB pieces: an upload never sits in memory whole.
    STREAM_CHUNK = 1024 * 1024

    @classmethod
    def save_stream(cls, user_token: str, folder_name: str, file_id: str, source,
                    max_bytes: int) -> tuple:
        """Copy a binary file-like ``source`` to storage in chunks, enforcing ``max_bytes``.

        The size limit is checked while copying: once more than ``max_bytes`` have been
        read, the partial file is deleted and ``FileTooLargeError`` is raised.

        Returns ``(relative_path, size, sha256 hex)``.
        """
        from db.filedb import FileTooLargeError, file_too_large_message
        folder_path = cls._get_folder_storage_path(user_token, folder_name)
        target = folder_path / file_id
        digest = hashlib.sha256()
        size = 0
        try:
            with open(target, 'wb') as out:
                while True:
                    chunk = source.read(cls.STREAM_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise FileTooLargeError(file_too_large_message(max_bytes))
                    digest.update(chunk)
                    out.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        relative_path = cls.relative_folder_path(user_token, folder_name) + str(file_id)
        logger.info(f"File saved successfully: {relative_path} ({size} bytes)")
        return relative_path, size, digest.hexdigest()

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
            folder_path = cls.folder_dir(user_token, folder_name)
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
        return cls.move_folder(user_token, old_folder_name, user_token, new_folder_name)

    @classmethod
    def move_folder(cls, old_owner: str, old_folder_name: str,
                    new_owner: str, new_folder_name: str) -> bool:
        """Move a folder's directory to another owner and/or name; return whether it moved.

        False when the source is missing, the target already exists or the move fails.
        Raises ValueError for an unsafe owner key or folder name.
        """
        try:
            _validate_safe_name(old_folder_name, "old_folder_name")
            _validate_safe_name(new_folder_name, "new_folder_name")
            old_path = cls.STORAGE_ROOT / owner_dir_name(old_owner) / old_folder_name
            new_path = cls.STORAGE_ROOT / owner_dir_name(new_owner) / new_folder_name

            if not old_path.exists():
                logger.warning(f"Folder not found for moving: {old_owner[:8]}.../{old_folder_name}")
                return False

            if new_path.exists():
                logger.error(f"Target folder already exists: {new_owner[:8]}.../{new_folder_name}")
                return False

            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)
            logger.info(f"Folder moved: {old_owner[:8]}.../{old_folder_name} -> "
                        f"{new_owner[:8]}.../{new_folder_name}")
            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to move folder {old_owner[:8]}.../{old_folder_name} to "
                         f"{new_owner[:8]}.../{new_folder_name}: {e}")
            return False
