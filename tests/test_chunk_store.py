"""Single-transaction replace of a file's chunks, with the search vector in the same INSERT."""

import json
from types import SimpleNamespace

from llama_index.core.schema import TextNode

from src.domain.rag import chunk_store as cs


class _Conn:
    def __init__(self, log, generated="s"):
        self.log, self.generated = log, generated

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.log.append((sql, params))
        if "pg_attribute" in sql:
            return SimpleNamespace(scalar=lambda: self.generated)
        if sql.startswith("DELETE"):
            return SimpleNamespace(rowcount=3)
        return SimpleNamespace(rowcount=len(params or []))

    def commit(self):
        self.log.append(("COMMIT", None))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, generated="s"):
        self.log, self.generated, self.tx = [], generated, 0

    def connect(self):
        return _Conn(self.log, self.generated)

    def begin(self):
        self.tx += 1
        self.log.append(("BEGIN", None))
        return _Conn(self.log, self.generated)


def _nodes():
    leaf = TextNode(id_="L1", text="36-month warranty.",
                    metadata={"file_id": "f", "node_role": "leaf", "breadcrumb": "Aurora › 5. Warranty"})
    leaf.embedding = [0.5, 0.25]
    parent = TextNode(id_="P1", text="36-month warranty.", metadata={"file_id": "f", "node_role": "parent"})
    parent.embedding = [0.0, 0.0]
    return [leaf, parent]


def test_replace_is_one_transaction_with_tsv_in_insert():
    eng = _Engine()
    store = SimpleNamespace(flat_metadata=False, use_jsonb=False, _initialize=lambda: None)
    n = cs.replace_file_chunks(store, "f", _nodes(), {"L1": "Aurora › 5. Warranty\n36-month warranty."},
                               "simple", segment=lambda xs: [x.upper() for x in xs],
                               engine=eng, table="data_1_x")
    assert n == 2 and eng.tx == 1
    sqls = [s for s, _ in eng.log]
    assert any("DROP EXPRESSION" in s for s in sqls)          # generated column detached first
    begin = sqls.index("BEGIN")
    assert sqls[begin + 1].startswith("DELETE") and sqls[begin + 2].startswith("INSERT")
    rows = eng.log[begin + 2][1]
    leaf, parent = rows
    assert leaf["text"] == "36-month warranty."                # displayed text = original text
    assert leaf["tsv"] == "AURORA › 5. WARRANTY\n36-MONTH WARRANTY."  # segmented retrieval text
    assert parent["tsv"] is None                               # parents are not keyword-searchable
    assert leaf["emb"] == "[0.5,0.25]" and leaf["cfg"] == "simple"
    meta = json.loads(leaf["meta"])
    assert meta["breadcrumb"] == "Aurora › 5. Warranty" and "_node_content" in meta


def test_already_detached_column_is_not_altered_again():
    eng = _Engine(generated="")
    store = SimpleNamespace(flat_metadata=False, use_jsonb=True)
    cs.replace_file_chunks(store, "f", _nodes(), {}, "english", engine=eng, table="t")
    sqls = [s for s, _ in eng.log]
    assert not any("DROP EXPRESSION" in s for s in sqls)
    assert any("CAST(:meta AS jsonb)" in s for s in sqls)


def test_empty_node_list_still_deletes_old_rows():
    eng = _Engine(generated="")
    store = SimpleNamespace(flat_metadata=False)
    assert cs.replace_file_chunks(store, "f", [], {}, engine=eng, table="t") == 0
    assert any(s.startswith("DELETE") for s, _ in eng.log)
    assert not any(s.startswith("INSERT") for s, _ in eng.log)


def test_fetch_reusable_parses_rows_and_vectors():
    class _C(_Conn):
        def execute(self, stmt, params=None):
            self.log.append((str(stmt), params))
            return SimpleNamespace(fetchall=lambda: [
                ("h1", "ctx", "k1", "e1", "[0.5,0.25]"),
                ("h2", None, None, "e1", "not a vector"),
            ])

    eng = SimpleNamespace(connect=lambda: _C([]))
    out = cs.fetch_reusable(None, "f", engine=eng, table="t")
    assert out["h1"] == {"context_prefix": "ctx", "ctx_key": "k1", "emb_key": "e1", "embedding": [0.5, 0.25]}
    assert out["h2"]["embedding"] is None and out["h2"]["context_prefix"] == ""


def test_fetch_reusable_missing_table_is_empty():
    def _boom():
        raise RuntimeError('relation "t" does not exist')

    assert cs.fetch_reusable(None, "f", engine=SimpleNamespace(connect=_boom), table="t") == {}
