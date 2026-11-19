"""Tags and metadata rules (src/domain/rag/doc_meta.py): every write path uses them."""

from __future__ import annotations

import pytest

from src.domain.rag import doc_meta as M
from src.domain.rag.doc_meta import DocMetaError


class TestTags:
    def test_trimmed_and_inner_whitespace_collapsed(self):
        assert M.normalize_tags(["  Q3   report\t ", "x"]) == ["Q3 report", "x"]

    def test_case_insensitive_duplicates_keep_first_spelling(self):
        assert M.normalize_tags(["Invoice", "invoice", "INVOICE ", "2024"]) == ["Invoice", "2024"]

    def test_empty_tags_dropped(self):
        assert M.normalize_tags(["", "   ", "a"]) == ["a"]

    def test_none_is_no_tags(self):
        assert M.normalize_tags(None) == []

    def test_max_length(self):
        assert M.normalize_tags(["x" * M.MAX_TAG_LEN]) == ["x" * M.MAX_TAG_LEN]
        with pytest.raises(DocMetaError, match="longer than 64"):
            M.normalize_tags(["x" * (M.MAX_TAG_LEN + 1)])

    def test_max_count_after_dedup(self):
        ok = [f"t{i}" for i in range(M.MAX_TAGS)]
        assert M.normalize_tags(ok + ["T0"]) == ok             # the duplicate does not count
        with pytest.raises(DocMetaError, match="at most 32 tags"):
            M.normalize_tags(ok + ["one more"])

    @pytest.mark.parametrize("bad", ["a,b", {"a": 1}, b"x"])
    def test_tags_must_be_a_list(self, bad):
        with pytest.raises(DocMetaError, match="list of strings"):
            M.normalize_tags(bad)

    @pytest.mark.parametrize("bad", [1, None, ["x"], {"a": 1}])
    def test_each_tag_must_be_a_string(self, bad):
        with pytest.raises(DocMetaError, match="must be a string"):
            M.normalize_tags([bad])

    def test_form_string(self):
        assert M.parse_tag_string(" invoice, 2024 ,,Invoice") == ["invoice", "2024"]
        assert M.parse_tag_string(None) == []

    @pytest.mark.parametrize("stored, out", [
        (None, []), ("x", []), (["a", 1, "b"], ["a", "b"]), (["A"], ["A"])])
    def test_stored_tags_tolerates_json_null(self, stored, out):
        assert M.stored_tags(stored) == out

    def test_tag_key_is_lowercase(self):
        assert M.tag_key("Ab C") == "ab c"


class TestMetadata:
    def test_valid_values_kept(self):
        meta = {"client": " Acme ", "year": 2024, "ratio": 0.5, "signed": True, "due": "2024-03-01"}
        assert M.normalize_metadata(meta) == {
            "client": "Acme", "year": 2024, "ratio": 0.5, "signed": True, "due": "2024-03-01"}

    def test_none_is_empty(self):
        assert M.normalize_metadata(None) == {}

    @pytest.mark.parametrize("key", [
        "Client", "with space", "a-b", "", "x" * 41, "a'; DROP TABLE \"Files\"; --",
        "a.b", "ключ", 5])
    def test_bad_keys_refused(self, key):
        with pytest.raises(DocMetaError, match="metadata key"):
            M.normalize_metadata({key: "v"})

    def test_key_limits(self):
        assert M.normalize_metadata({"a" * 40: 1}) == {"a" * 40: 1}
        many = {f"k{i}": i for i in range(M.MAX_META_KEYS)}
        assert M.normalize_metadata(many) == many
        with pytest.raises(DocMetaError, match="at most 32 metadata keys"):
            M.normalize_metadata({**many, "extra": 1})

    @pytest.mark.parametrize("value", [{"a": 1}, [1, 2], ("a",)])
    def test_nested_values_refused_with_a_clear_message(self, value):
        with pytest.raises(DocMetaError, match="nested values are not allowed"):
            M.normalize_metadata({"k": value})

    def test_null_refused(self):
        with pytest.raises(DocMetaError, match="null is not a value"):
            M.normalize_metadata({"k": None})

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_numbers_must_be_finite(self, value):
        with pytest.raises(DocMetaError, match="finite"):
            M.normalize_metadata({"k": value})

    def test_string_length(self):
        assert M.normalize_metadata({"k": "x" * 500}) == {"k": "x" * 500}
        with pytest.raises(DocMetaError, match="longer than 500"):
            M.normalize_metadata({"k": "x" * 501})

    def test_impossible_date_refused(self):
        with pytest.raises(DocMetaError, match="not a valid date"):
            M.normalize_metadata({"due": "2024-02-30"})

    def test_other_types_refused(self):
        with pytest.raises(DocMetaError, match="unsupported value type"):
            M.normalize_metadata({"k": object()})

    def test_must_be_an_object(self):
        with pytest.raises(DocMetaError, match="must be an object"):
            M.normalize_metadata(["a"])

    def test_json_form_field(self):
        assert M.parse_metadata_json('{"year": 2024}') == {"year": 2024}
        assert M.parse_metadata_json("  ") == {} and M.parse_metadata_json(None) == {}
        with pytest.raises(DocMetaError, match="not valid JSON"):
            M.parse_metadata_json("{year: 2024}")
        with pytest.raises(DocMetaError, match="nested"):
            M.parse_metadata_json('{"a": {"b": 1}}')

    @pytest.mark.parametrize("value, kind", [
        (True, "boolean"), (3, "number"), (2.5, "number"), ("2024-01-31", "date"),
        ("2024-02-31", "string"), ("text", "string")])
    def test_value_type(self, value, kind):
        assert M.value_type(value) == kind

    def test_stored_metadata(self):
        assert M.stored_metadata(None) == {} and M.stored_metadata({"a": 1}) == {"a": 1}
