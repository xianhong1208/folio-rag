"""The document filter (src/domain/rag/doc_filter.py): parsing, the SQL it builds and
the owner scope it always adds. The SQL itself runs in tests/integration/test_doc_filter_db.py."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.domain.rag import doc_filter as DF
from src.domain.rag.doc_filter import ALL_FOLDERS, DocumentFilter, build_where


def F(**kw) -> DocumentFilter:
    return DocumentFilter.model_validate(kw)


def where(flt=None, owner="sub:me", **kw):
    return build_where(flt, owner=owner, **kw)


class TestParse:
    def test_empty(self):
        f = F()
        assert f.is_empty() and DF.parse_filter(None) is None
        assert DF.parse_filter({}).is_empty()
        assert DF.parse_filter(f) is f

    def test_full_filter(self):
        f = F(tags_any=[" Invoice "], tags_all=["Signed"], folders=["hr", 3], file_type=".PDF",
              uploaded_after="2024-01-01", uploaded_before="2025-01-01T00:00:00+08:00",
              name_contains="q3", metadata={"client": "Acme", "year": {"gte": 2024, "lte": 2025},
                                            "lang": {"in": ["zh", "en"]}, "x": {"exists": False}})
        assert f.tags_any == ["Invoice"] and f.file_type == ["pdf"] and not f.is_empty()
        ops = [(c.key, c.op, c.value) for c in f.meta_conditions()]
        assert ("client", "eq", "Acme") in ops and ("year", "gte", 2024) in ops
        assert ("year", "lte", 2025) in ops and ("lang", "in", ("zh", "en")) in ops
        assert ("x", "exists", False) in ops

    @pytest.mark.parametrize("ids", [[], ["nope"], ["1; DROP TABLE x"], [12]])
    def test_bad_file_ids(self, ids):
        with pytest.raises(ValidationError):
            F(file_ids=ids)

    def test_file_ids_not_empty(self):
        f = F(file_ids=["00000000-0000-4000-8000-000000000001"])
        assert not f.is_empty()
        assert "file_ids" in f.fingerprint()

    def test_unknown_field_is_named(self):
        with pytest.raises(ValidationError) as e:
            F(tag="x")
        assert DF.filter_error_detail(e.value) == "invalid filter: tag: unknown filter field"

    @pytest.mark.parametrize("cond, match", [
        ({"year": {"gt": 1}}, "unknown operator\\(s\\) gt"),
        ({"year": {"$ne": 1, "in": [1]}}, "unknown operator\\(s\\) \\$ne"),
        ({"year": {}}, "empty condition"),
        ({"year": {"in": []}}, "'in' takes a list"),
        ({"year": {"in": "a"}}, "'in' takes a list"),
        ({"year": {"exists": "yes"}}, "'exists' takes true or false"),
        ({"year": {"gte": "soon"}}, "'gte' takes a number or a YYYY-MM-DD date"),
        ({"year": {"lte": True}}, "'lte' takes a number"),
        ({"year": [1, 2]}, "a list is not a value"),
        ({"year": None}, "null is not a value"),
        ({"Bad Key": 1}, "metadata key"),
        ({"k": {"in": [{"a": 1}]}}, "nested values"),
    ])
    def test_bad_metadata_conditions(self, cond, match):
        with pytest.raises(ValidationError, match=match):
            F(metadata=cond)

    @pytest.mark.parametrize("ft", ["p%df", "../x", "application/pdf;x", "*/*", "a" * 17, 3])
    def test_bad_file_types(self, ft):
        with pytest.raises(ValidationError):
            F(file_type=[ft])

    def test_mime_and_extensions(self):
        assert F(file_type=["Application/PDF", "image/*", ".docx"]).file_type == [
            "application/pdf", "image/*", "docx"]

    @pytest.mark.parametrize("v", ["yesterday", "2024-13-01", "2024/01/01"])
    def test_bad_dates(self, v):
        with pytest.raises(ValidationError, match="uploaded_after"):
            F(uploaded_after=v)

    def test_tag_too_long(self):
        with pytest.raises(ValidationError, match="longer than 64"):
            F(tags_any=["x" * 65])

    def test_too_many_conditions(self):
        with pytest.raises(ValidationError, match="at most 100"):
            F(metadata={f"k{i}": i for i in range(101)})

    def test_fingerprint_is_canonical(self):
        a = F(tags_any=["a"], metadata={"y": 1, "x": "b"})
        b = F(metadata={"x": "b", "y": 1}, tags_any=["a"])
        assert a.fingerprint() == b.fingerprint() != F(tags_any=["b"]).fingerprint()

    def test_error_detail_without_errors(self):
        assert DF.filter_error_detail(ValueError("boom")) == "boom"


class TestOwnerScope:
    def test_owner_always_applied(self):
        sql, params = where(F(folders=["other-persons-folder", 99]))
        assert sql.startswith('f.folder_id IN (SELECT d.id FROM "Folders" d WHERE d.user_token = :owner_0)')
        assert params["owner_0"] == "sub:me"

    def test_no_filter_still_scoped(self):
        sql, params = where(None)
        assert ":owner_0" in sql and params == {"owner_0": "sub:me"}

    @pytest.mark.parametrize("owner", [None, "", 0])
    def test_owner_required(self, owner):
        with pytest.raises(ValueError, match="owner key"):
            build_where(F(), owner=owner)

    def test_open_mode_sentinel(self):
        assert build_where(None, owner=ALL_FOLDERS) == ("TRUE", {})
        assert DF.owner_scope(None) is ALL_FOLDERS and DF.owner_scope("sub:a") == "sub:a"

    def test_folder_id(self):
        sql, params = where(None, folder_id=7)
        assert "f.folder_id = :folder_1" in sql and params["folder_1"] == 7


class TestSql:
    INJECTION = "x'); DELETE FROM \"Files\"; --"

    def test_values_never_reach_the_sql_text(self):
        flt = F(tags_any=[self.INJECTION], tags_all=[self.INJECTION], folders=[self.INJECTION],
                name_contains=self.INJECTION, metadata={"k": self.INJECTION,
                                                        "j": {"in": [self.INJECTION]}})
        sql, params = where(flt)
        assert "DELETE" not in sql and "x')" not in sql
        assert any(self.INJECTION.lower() in json.dumps(v) or v == [self.INJECTION]
                   for v in params.values())

    def test_metadata_keys_are_bound_not_pasted(self):
        sql, params = where(F(metadata={"secret_key": {"gte": 1}, "other_key": {"exists": True}}))
        assert "secret_key" not in sql and "other_key" not in sql
        assert "secret_key" in params.values() and "other_key" in params.values()

    def test_tags_any_lowercased_and_bound(self):
        sql, params = where(F(tags_any=["Invoice", "INVOICE", "Tax"]))
        assert "lower(t.v) = ANY(:tags_any_1)" in sql
        assert params["tags_any_1"] == ["invoice", "tax"]
        assert "jsonb_typeof(f.tags) = 'array'" in sql             # JSON null reads as no tags

    def test_tags_all_one_exists_per_tag(self):
        sql, params = where(F(tags_all=["A", "b", "B"]))
        assert sql.count("EXISTS") == 2
        assert sorted(v for k, v in params.items() if k.startswith("tag_all")) == ["a", "b"]

    def test_folders_by_name_and_id(self):
        sql, params = where(F(folders=["hr", 5]))
        assert params["folder_ids_1"] == [5] and params["folder_names_2"] == ["hr"]

    def test_file_types(self):
        sql, params = where(F(file_type=["pdf", "application/msword", "image/*", "md"]))
        vals = list(params.values())[1:]
        assert vals == ["%.pdf", "application/msword", "image/%", "%.md"]
        assert sql.count("ESCAPE") == 3

    def test_empty_file_type_matches_nothing(self):
        sql, _ = where(F(file_type=[]))
        assert "(FALSE)" in sql

    def test_dates(self):
        sql, params = where(F(uploaded_after="2024-01-01", uploaded_before="2024-02-01T08:00:00+08:00"))
        assert "f.upload_time >= :after_1" in sql and "f.upload_time < :before_2" in sql
        assert params["after_1"] == datetime(2024, 1, 1)
        assert params["before_2"] == datetime(2024, 2, 1, 0, 0)         # converted to UTC

    def test_file_ids_are_bound_uuids(self):
        a = "2F0C9A4E-0000-4000-8000-000000000001"
        sql, params = where(F(file_ids=[a]))
        assert "f.id = ANY(CAST(:file_ids_1 AS uuid[]))" in sql
        assert params["file_ids_1"] == [a.lower()]
        assert a.lower() not in sql

    def test_name_contains_escapes_like_wildcards(self):
        _, params = where(F(name_contains="50%_off\\"))
        assert params["name_1"] == "%50\\%\\_off\\\\%"

    def test_metadata_equality_uses_containment(self):
        sql, params = where(F(metadata={"client": "Acme", "year": 2024, "ok": True}))
        assert sql.count("f.metadata @> CAST(") == 3
        assert json.loads(params["meta_1"]) == {"client": "Acme"}
        assert json.loads(params["meta_2"]) == {"year": 2024}

    def test_metadata_in(self):
        sql, params = where(F(metadata={"lang": {"in": ["zh", "en"]}}))
        assert " OR " in sql and json.loads(params["meta_2"]) == {"lang": "en"}

    def test_metadata_ranges(self):
        sql, params = where(F(metadata={"year": {"gte": 2024}, "due": {"lte": "2024-12-31"}}))
        assert "CAST(f.metadata ->> :meta_key_1 AS numeric) >= :meta_num_2" in sql
        assert "jsonb_typeof(f.metadata -> :meta_key_3) = 'string'" in sql
        assert "COLLATE \"C\" <= :meta_date_4" in sql
        assert "'^[0-9]{4}-[0-9]{2}-[0-9]{2}$'" in sql

    def test_metadata_exists(self):
        sql, _ = where(F(metadata={"a": {"exists": True}, "b": {"exists": False}}))
        assert "jsonb_exists(" in sql and "NOT jsonb_exists(" in sql


class _Conn:
    """A sync connection stub: records statements, answers from a queue."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), dict(params or {})))
        res = self.results.pop(0)
        return SimpleNamespace(fetchall=lambda: res, scalar=lambda: res,
                               mappings=lambda: SimpleNamespace(fetchall=lambda: res))


class TestQueries:
    def test_matching_file_ids(self):
        conn = _Conn([("a",), ("b",)])
        assert DF.matching_file_ids(conn, F(tags_any=["x"]), owner="sub:me", folder_id=3) == ["a", "b"]
        sql, params = conn.calls[0]
        assert ":owner_0" in sql and params["folder_1"] == 3

    def test_search_documents_page(self):
        row = {"id": "u1", "file_name": "a.pdf", "file_size": 10, "mime_type": "application/pdf",
               "upload_time": datetime(2024, 1, 2), "updated_time": None, "tags": None,
               "metadata": {"y": 1}, "description": "d", "folder_id": 4, "folder_name": "hr",
               "index_status": None}
        conn = _Conn(7, [row])
        items, total = DF.search_documents(conn, None, owner="sub:me", sort="name", order="asc",
                                           limit=500, offset=-3)
        assert total == 7
        assert items == [{"id": "u1", "name": "a.pdf", "folder": {"id": 4, "name": "hr"},
                          "size_bytes": 10, "mime": "application/pdf",
                          "uploaded_at": "2024-01-02T00:00:00", "updated_at": None,
                          "status": "unindexed", "tags": [], "metadata": {"y": 1},
                          "description": "d"}]
        sql, params = conn.calls[1]
        assert "ORDER BY lower(f.file_name) ASC, f.id ASC" in sql
        assert params["_limit"] == DF.MAX_PAGE and params["_offset"] == 0
        assert params["owner_0"] == "sub:me"

    @pytest.mark.parametrize("kw, msg", [(dict(sort="x"), "sort"), (dict(order="up"), "order")])
    def test_search_documents_rejects_unknown_sort(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            DF.search_documents(_Conn(), None, owner="sub:me", **kw)

    def test_tag_counts(self):
        conn = _Conn([("Invoice", 3), ("tax", 1)])
        assert DF.tag_counts(conn, owner="sub:me") == [
            {"tag": "Invoice", "count": 3}, {"tag": "tax", "count": 1}]
        sql, params = conn.calls[0]
        assert "GROUP BY lower(t.v)" in sql and params == {"owner_0": "sub:me"}

    def test_metadata_keys(self):
        conn = _Conn([("due", "date", 2), ("year", "number", 4), ("year", "string", 1)])
        assert DF.metadata_keys(conn, owner="sub:me") == [
            {"key": "year", "count": 5, "types": {"number": 4, "string": 1}},
            {"key": "due", "count": 2, "types": {"date": 2}}]

    def test_resolve_file_ids(self):
        assert DF.resolve_file_ids(None, owner="sub:me") is None
        assert DF.resolve_file_ids(F(), owner="sub:me") is None
        conn = _Conn([("b",), ("a",)])
        cm = MagicMock()
        cm.__enter__.return_value = conn
        engine = MagicMock()
        engine.connect.return_value = cm
        with patch("db.db.get_engine", return_value=engine):
            assert DF.resolve_file_ids(F(tags_any=["x"]), owner="sub:me", folder_id=2) == ("a", "b")
        assert conn.calls[0][1]["folder_1"] == 2
