"""The backup archive format: names, manifest validation and defensive extraction."""

import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.backup import archive as A
from src.backup.archive import ArchiveError


def _manifest(checksums):
    return {"format": A.FORMAT, "format_version": A.FORMAT_VERSION, "created_at": "2026-11-14T00:00:00+00:00",
            "folio_version": "1.4.2", "embedding": {"model": "m", "dimension": 8}, "checksums": checksums}


def _good_archive(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "dump").write_bytes(b"PGDMP fake dump")
    (src / "a.txt").write_text("hello", encoding="utf-8")
    w = A.ArchiveWriter(tmp_path / "out.tar")
    w.add_file(A.DUMP_NAME, src / "dump")
    w.add_file("storage/sub_u/docs/a.txt", src / "a.txt")
    w.add_manifest(_manifest(dict(w.checksums)))
    return w.commit()


def _raw_archive(path: Path, entries):
    """entries: (name, bytes | None for a symlink to /etc/passwd)."""
    with tarfile.open(path, "w") as t:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                t.addfile(info)
            else:
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
    return path


def _sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


class TestNames:
    def test_archive_name_round_trips_through_archive_time(self):
        now = datetime(2026, 11, 14, 3, 50, 2, tzinfo=timezone.utc)
        name = A.archive_name(now)
        assert name == "folio-backup-20261114T035002Z.tar"
        assert A.archive_time(name) == now

    @pytest.mark.parametrize("name", ["mine.tar", "folio-backup-2026.tar", "folio-backup-20261114T035002Z.tar.partial"])
    def test_other_names_have_no_time(self, name):
        assert A.archive_time(name) is None

    @pytest.mark.parametrize("name", ["manifest.json", "database.dump", "storage/x", "storage/sub_a/f/b c.pdf"])
    def test_safe_names(self, name):
        assert A.safe_member_path(name) == name

    @pytest.mark.parametrize("name", ["", None, "/etc/passwd", "storage/../x", "../x", "storage/", "storage//x",
                                      "storage/./x", "a\\b", "C:/x", "other.txt", "storage", "storage/x\x00"])
    def test_unsafe_names_are_refused(self, name):
        with pytest.raises(ArchiveError):
            A.safe_member_path(name)


def test_absolute_paths_are_named_as_such():
    with pytest.raises(ArchiveError, match="absolute"):
        A.safe_member_path("/etc/passwd")
    with pytest.raises(ArchiveError, match="absolute"):
        A.safe_member_path("C:/storage/x")


class TestFixedSize:
    def test_a_file_that_shrank_is_padded(self):
        r = A._FixedSize(io.BytesIO(b"abc"), 5)
        assert r.read(10) == b"abc\0\0" and r.read() == b""

    def test_a_file_that_grew_is_cut(self):
        r = A._FixedSize(io.BytesIO(b"abcdef"), 4)
        assert r.read() == b"abcd" and r.read(1) == b""


class TestRoundTrip:
    def test_write_read_verify_extract(self, tmp_path):
        path = _good_archive(tmp_path)
        assert not path.with_name("out.tar.partial").exists()
        m = A.read_manifest(path)
        assert set(m["checksums"]) == {"database.dump", "storage/sub_u/docs/a.txt"}
        assert A.verify_and_extract(path, m) == {"entries": 2, "bytes": len(b"PGDMP fake dump") + 5}
        dest = tmp_path / "dest"
        dest.mkdir()
        A.verify_and_extract(path, m, dest)
        assert (dest / "storage/sub_u/docs/a.txt").read_text() == "hello"
        assert (dest / "database.dump").read_bytes() == b"PGDMP fake dump"

    def test_abort_removes_the_partial_file(self, tmp_path):
        w = A.ArchiveWriter(tmp_path / "x.tar")
        assert w.partial.exists()
        w.abort()
        assert not w.partial.exists() and not (tmp_path / "x.tar").exists()
        w.abort()  # twice is fine


class TestValidateManifest:
    @pytest.mark.parametrize("mutate,msg", [
        (lambda m: "x", "not an object"),
        (lambda m: {**m, "format": "zip"}, "not a Folio backup"),
        (lambda m: {**m, "format_version": 0}, "format_version"),
        (lambda m: {**m, "format_version": True}, "format_version"),
        (lambda m: {**m, "format_version": A.FORMAT_VERSION + 1}, "newer Folio"),
        (lambda m: {**m, "checksums": {}}, "no checksums"),
        (lambda m: {**m, "checksums": {**m["checksums"], "storage/x": "zz"}}, "invalid checksum"),
        (lambda m: {**m, "checksums": {**m["checksums"], "storage/../x": "0" * 64}}, "traversal"),
        (lambda m: {**m, "checksums": {**m["checksums"], "manifest.json": "0" * 64}}, "its own"),
        (lambda m: {k: v for k, v in m.items() if k != "folio_version"}, "folio_version"),
        (lambda m: {**m, "embedding": {"model": "m"}}, "embedding"),
    ])
    def test_bad_manifests_are_refused(self, mutate, msg):
        with pytest.raises(ArchiveError, match=msg):
            A.validate_manifest(mutate(_manifest({"database.dump": "0" * 64})))


class TestReadManifest:
    def test_not_a_tar(self, tmp_path):
        p = tmp_path / "x.tar"
        p.write_bytes(b"definitely not a tar" * 100)
        with pytest.raises(ArchiveError, match="tar"):
            A.read_manifest(p)

    def test_no_manifest(self, tmp_path):
        p = _raw_archive(tmp_path / "x.tar", [("database.dump", b"x")])
        with pytest.raises(ArchiveError, match="no manifest"):
            A.read_manifest(p)

    def test_manifest_twice(self, tmp_path):
        data = json.dumps(_manifest({"database.dump": _sha(b"x")})).encode()
        p = _raw_archive(tmp_path / "x.tar", [("manifest.json", data), ("manifest.json", data)])
        with pytest.raises(ArchiveError, match="twice"):
            A.read_manifest(p)

    def test_manifest_not_json(self, tmp_path):
        p = _raw_archive(tmp_path / "x.tar", [("manifest.json", b"{nope")])
        with pytest.raises(ArchiveError, match="JSON"):
            A.read_manifest(p)

    def test_manifest_is_a_link(self, tmp_path):
        p = _raw_archive(tmp_path / "x.tar", [("manifest.json", None)])
        with pytest.raises(ArchiveError, match="regular file"):
            A.read_manifest(p)


class TestVerifyRefuses:
    def _check(self, tmp_path, entries, checksums, match):
        m = _manifest(checksums)
        data = json.dumps(m).encode()
        p = _raw_archive(tmp_path / "x.tar", entries + [("manifest.json", data)])
        dest = tmp_path / "dest"
        dest.mkdir()
        with pytest.raises(ArchiveError, match=match):
            A.verify_and_extract(p, m, dest)
        return dest

    def test_an_entry_escaping_the_restore_dir(self, tmp_path):
        dest = self._check(tmp_path, [("database.dump", b"x"), ("../../evil.txt", b"x")],
                           {"database.dump": _sha(b"x")}, "traversal")
        assert not (tmp_path / "evil.txt").exists() and not (dest.parent.parent / "evil.txt").exists()

    def test_an_unlisted_entry(self, tmp_path):
        self._check(tmp_path, [("database.dump", b"x"), ("storage/extra", b"y")],
                    {"database.dump": _sha(b"x")}, "not listed")

    def test_a_symlink(self, tmp_path):
        self._check(tmp_path, [("database.dump", b"x"), ("storage/link", None)],
                    {"database.dump": _sha(b"x"), "storage/link": "0" * 64}, "not a regular file")

    def test_a_duplicate(self, tmp_path):
        self._check(tmp_path, [("database.dump", b"x"), ("database.dump", b"x")],
                    {"database.dump": _sha(b"x")}, "twice")

    def test_a_changed_file(self, tmp_path):
        self._check(tmp_path, [("database.dump", b"x"), ("storage/a", b"changed")],
                    {"database.dump": _sha(b"x"), "storage/a": _sha(b"original")}, "checksum mismatch")

    def test_a_missing_file(self, tmp_path):
        self._check(tmp_path, [("database.dump", b"x")],
                    {"database.dump": _sha(b"x"), "storage/a": _sha(b"a")}, "missing")

    def test_no_manifest_entry(self, tmp_path):
        m = _manifest({"database.dump": _sha(b"x")})
        p = _raw_archive(tmp_path / "x.tar", [("database.dump", b"x")])
        with pytest.raises(ArchiveError, match="no manifest"):
            A.verify_and_extract(p, m)

    def test_not_a_tar(self, tmp_path):
        p = tmp_path / "x.tar"
        p.write_bytes(b"\x00garbage" * 1000)
        with pytest.raises(ArchiveError):
            A.verify_and_extract(p, _manifest({"database.dump": "0" * 64}))
