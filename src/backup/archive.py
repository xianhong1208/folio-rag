"""The backup archive: an uncompressed tar with a manifest, a database dump and storage/.

Layout (``folio-backup-<UTC time>.tar``)::

    database.dump        pg_dump --format=custom (already compressed)
    storage/...          the files under storage/, paths relative to it
    manifest.json        written last; what the archive was made from + a SHA-256 per entry

Reading is defensive because an archive may come from anywhere: every entry name is
checked (no absolute paths, no ``..``, no backslashes, only the three kinds above),
only regular files are accepted (no links, devices or directories that point
elsewhere), every entry must be listed in the manifest with a matching SHA-256, and
nothing listed may be missing. Nothing is extracted with ``tarfile.extractall``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Optional

FORMAT = "folio-backup"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
DUMP_NAME = "database.dump"
STORAGE_PREFIX = "storage/"

# Names the server's schedule (and `main.py backup` without --output) gives archives;
# only these are ever pruned.
ARCHIVE_RE = re.compile(r"^folio-backup-(\d{8}T\d{6}Z)\.tar$")
_TIME_FORMAT = "%Y%m%dT%H%M%SZ"

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024


class ArchiveError(Exception):
    """The archive is unreadable, incomplete or has been tampered with."""


def archive_name(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"folio-backup-{now.astimezone(timezone.utc).strftime(_TIME_FORMAT)}.tar"


def archive_time(name: str) -> Optional[datetime]:
    """When a scheduled-style archive was made (from its name), or None for other names."""
    m = ARCHIVE_RE.match(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), _TIME_FORMAT).replace(tzinfo=timezone.utc)


def safe_member_path(name: Any) -> str:
    """Validate an entry name and return it; raise ArchiveError for anything unsafe.

    Allowed: ``manifest.json``, ``database.dump`` and ``storage/<relative path>``.
    """
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ArchiveError(f"unsafe entry name in archive: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ArchiveError(f"absolute path in archive: {name!r}")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ArchiveError(f"path traversal or empty path segment in archive: {name!r}")
    if name in (MANIFEST_NAME, DUMP_NAME):
        return name
    if name.startswith(STORAGE_PREFIX) and len(parts) >= 2:
        return name
    raise ArchiveError(f"unexpected entry in archive: {name!r}")


class _HashingReader:
    """File wrapper that hashes what tarfile reads from it."""

    def __init__(self, f: BinaryIO):
        self._f = f
        self.sha = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        data = self._f.read(n)
        self.sha.update(data)
        return data


class _FixedSize:
    """Reads exactly ``size`` bytes: a file that shrinks while it is archived is padded
    with zeros and one that grows is cut, so the tar stream always stays well-formed."""

    def __init__(self, f: BinaryIO, size: int):
        self._f, self._left = f, size

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        n = self._left if n is None or n < 0 else min(n, self._left)
        data = self._f.read(n)
        if len(data) < n:
            data += b"\0" * (n - len(data))
        self._left -= len(data)
        return data


class ArchiveWriter:
    """Write an archive to ``<path>.partial`` and rename it into place on commit."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.checksums: Dict[str, str] = {}
        self.sizes: Dict[str, int] = {}
        self._tar = tarfile.open(self.partial, "w", format=tarfile.PAX_FORMAT)

    def add_file(self, arcname: str, src: Path) -> str:
        with open(src, "rb") as f:
            return self.add_open_file(arcname, f)

    def add_open_file(self, arcname: str, f: BinaryIO) -> str:
        """Add an open file under ``arcname``; return its SHA-256.

        The size is taken once (fstat) and exactly that many bytes are written (see
        _FixedSize), so a file that changes while it is read never breaks the tar.
        """
        arcname = safe_member_path(arcname)
        st = os.fstat(f.fileno())
        info = tarfile.TarInfo(arcname)
        info.size = st.st_size
        info.mtime = int(st.st_mtime)
        info.mode = 0o644
        reader = _HashingReader(_FixedSize(f, st.st_size))
        self._tar.addfile(info, reader)
        digest = reader.sha.hexdigest()
        self.checksums[arcname] = digest
        self.sizes[arcname] = st.st_size
        return digest

    def add_manifest(self, manifest: Dict[str, Any]) -> None:
        data = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(data)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        info.mode = 0o644
        self._tar.addfile(info, io.BytesIO(data))

    def commit(self) -> Path:
        self._tar.close()
        os.replace(self.partial, self.path)
        return self.path

    def abort(self) -> None:
        try:
            self._tar.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.partial.unlink()
        except FileNotFoundError:
            pass


def validate_manifest(m: Any) -> Dict[str, Any]:
    """Check the manifest's shape; raise ArchiveError when it is not one Folio can read."""
    if not isinstance(m, dict):
        raise ArchiveError("manifest.json is not an object")
    if m.get("format") != FORMAT:
        raise ArchiveError("manifest.json is not a Folio backup manifest")
    version = m.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ArchiveError("manifest.json has no valid format_version")
    if version > FORMAT_VERSION:
        raise ArchiveError(
            f"this archive uses backup format {version}; this Folio reads format "
            f"{FORMAT_VERSION} or older. Restore it with a newer Folio.")
    checksums = m.get("checksums")
    if not isinstance(checksums, dict) or DUMP_NAME not in checksums:
        raise ArchiveError("manifest.json lists no checksums for the database dump")
    for name, digest in checksums.items():
        safe_member_path(name)
        if name == MANIFEST_NAME:
            raise ArchiveError("manifest.json cannot list its own checksum")
        if not isinstance(digest, str) or not _HEX64.match(digest):
            raise ArchiveError(f"manifest.json has an invalid checksum for {name!r}")
    for key in ("created_at", "folio_version", "embedding"):
        if key not in m:
            raise ArchiveError(f"manifest.json has no {key!r}")
    emb = m.get("embedding")
    if not isinstance(emb, dict) or not isinstance(emb.get("dimension"), int):
        raise ArchiveError("manifest.json has no embedding model / dimension")
    return m


def read_manifest(path: Path) -> Dict[str, Any]:
    """Read and validate an archive's manifest (raises ArchiveError)."""
    try:
        with tarfile.open(path, "r:") as tar:
            found = None
            for member in tar:
                if member.name == MANIFEST_NAME:
                    if found is not None:
                        raise ArchiveError("the archive holds manifest.json twice")
                    found = member
            if found is None:
                raise ArchiveError("the archive has no manifest.json (not a Folio backup)")
            if not found.isfile() or found.size > MAX_MANIFEST_BYTES:
                raise ArchiveError("manifest.json is not a regular file of a sane size")
            f = tar.extractfile(found)
            raw = f.read() if f is not None else b""
    except tarfile.TarError as e:
        raise ArchiveError(f"not a readable tar archive: {e}") from e
    try:
        m = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ArchiveError(f"manifest.json is not valid JSON: {e}") from e
    return validate_manifest(m)


def verify_and_extract(path: Path, manifest: Dict[str, Any],
                       dest: Optional[Path] = None) -> Dict[str, int]:
    """Check every entry against the manifest; with ``dest``, also extract them there.

    Returns {"entries": n, "bytes": total}. Raises ArchiveError on an unsafe or unlisted
    entry, a non-regular file, a duplicate, a checksum mismatch or a missing entry.
    """
    checksums: Dict[str, str] = manifest["checksums"]
    seen = set()
    total = 0
    dest_root = Path(dest).resolve() if dest is not None else None
    try:
        with tarfile.open(path, "r:") as tar:
            for member in tar:
                name = safe_member_path(member.name)
                if not member.isfile():
                    raise ArchiveError(f"{name!r} is not a regular file (links and devices are refused)")
                if name in seen:
                    raise ArchiveError(f"{name!r} appears twice in the archive")
                seen.add(name)
                if name == MANIFEST_NAME:
                    continue
                expected = checksums.get(name)
                if expected is None:
                    raise ArchiveError(f"{name!r} is not listed in manifest.json")
                src = tar.extractfile(member)
                if src is None:  # pragma: no cover - isfile() above guarantees a stream
                    raise ArchiveError(f"cannot read {name!r}")
                sha = hashlib.sha256()
                out = None
                if dest_root is not None:
                    target = (dest_root / PurePosixPath(name)).resolve()
                    if dest_root not in target.parents:
                        raise ArchiveError(f"{name!r} would be written outside the restore directory")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    out = open(target, "wb")
                try:
                    while True:
                        chunk = src.read(_CHUNK)
                        if not chunk:
                            break
                        sha.update(chunk)
                        if out is not None:
                            out.write(chunk)
                finally:
                    if out is not None:
                        out.close()
                if sha.hexdigest() != expected:
                    raise ArchiveError(f"checksum mismatch for {name!r}: the archive is damaged or was changed")
                total += member.size
    except tarfile.TarError as e:
        raise ArchiveError(f"not a readable tar archive: {e}") from e
    missing = sorted(set(checksums) - seen)
    if missing:
        raise ArchiveError(f"{len(missing)} entr{'y' if len(missing) == 1 else 'ies'} listed in "
                           f"manifest.json {'is' if len(missing) == 1 else 'are'} missing, e.g. {missing[0]!r}")
    if MANIFEST_NAME not in seen:
        raise ArchiveError("the archive has no manifest.json")
    return {"entries": len(seen) - 1, "bytes": total}
