"""Incremental re-indexing: unchanged chunks keep their prefix and embedding."""

from types import SimpleNamespace

import pytest
from llama_index.core import Document

from src.domain.rag import hierarchical_indexer as hi
from src.domain.rag.chunk_identity import chunk_hash, embedding_key, leaf_node_id, parent_node_id


class _Gen:
    cache_key = "llm|prompt=1"
    max_doc_chars = 60000

    def __init__(self):
        self.calls = []
        self.kwargs = {}

    async def generate_prefixes(self, chunks, full_document, file_name, **kw):
        self.calls.extend(chunks)
        self.kwargs = kw
        return [f"ctx:{c[:12]}" for c in chunks]


class _Embed:
    model_name = "emb-1"

    def __init__(self):
        self.texts = []


class _Store:
    """In-memory stand-in for the folder table (fetch_reusable / replace_file_chunks)."""

    def __init__(self):
        self.rows = {}

    def fetch(self, vector_store, file_id):
        out = {}
        for n in self.rows.get(file_id, []):
            if n.metadata.get("node_role") == "leaf":
                out[n.metadata["chunk_hash"]] = {
                    "context_prefix": n.metadata.get("context_prefix", ""),
                    "ctx_key": n.metadata.get("ctx_key"),
                    "emb_key": n.metadata.get("emb_key"),
                    "embedding": n.embedding,
                }
        return out

    def replace(self, vector_store, file_id, nodes, search_texts, cfg, segment):
        self.rows[file_id] = list(nodes)
        self.search_texts = search_texts
        return len(nodes)


@pytest.fixture
def env(monkeypatch):
    store, embed = _Store(), _Embed()

    async def _embed_batch(model, texts, *a, **k):
        embed.texts.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]

    monkeypatch.setattr(hi, "fetch_reusable", store.fetch)
    monkeypatch.setattr(hi, "replace_file_chunks", store.replace)
    monkeypatch.setattr(hi, "_embed_batch_with_retry", _embed_batch)
    monkeypatch.setattr(hi.HierarchicalIndexer, "_text_search_config", staticmethod(lambda: "english"))
    monkeypatch.setattr("src.utils.db_bootstrap.ensure_hnsw_for_table", lambda *a: False)
    monkeypatch.setattr("src.domain.rag.chunk_lookup.ChunkLookup._get_table_name", staticmethod(lambda vs: "t"))
    monkeypatch.setattr("src.domain.rag.chunk_lookup.ChunkLookup._get_engine", staticmethod(lambda vs: None))
    gen = _Gen()
    indexer = hi.HierarchicalIndexer(leaf_chunk_size=256, context_generator=gen, embed_model=embed)
    return SimpleNamespace(store=store, embed=embed, gen=gen, indexer=indexer)


def _doc(warranty="36 months"):
    chunks = [
        {"text": "The router forms a mesh network.", "headings": ["Aurora", "1. Overview"]},
        {"text": f"The hardware warranty is {warranty}.", "headings": ["Aurora", "5. Warranty"]},
        {"text": "Idle power is 11 W.", "headings": ["Aurora", "6. Power"]},
    ]
    return Document(text="full text", metadata={"file_id": "F1", "file_name": "Aurora.docx",
                                                "_docling_chunks": chunks})


async def _index(env, doc):
    env.gen.calls.clear()
    env.embed.texts.clear()
    return await env.indexer.index_document(doc, vector_store=object())


async def test_first_index_then_unchanged_then_one_section_edit(env):
    first = await _index(env, _doc())
    assert (first["llm_calls"], first["embedded"]) == (3, 3)
    ids_before = {n.metadata["chunk_hash"]: n.node_id for n in env.store.rows["F1"]
                  if n.metadata["node_role"] == "leaf"}

    same = await _index(env, _doc())
    assert (same["llm_calls"], same["embedded"]) == (0, 0)
    assert env.gen.calls == [] and env.embed.texts == []

    edited = await _index(env, _doc("48 months"))
    assert (edited["llm_calls"], edited["embedded"]) == (1, 1)
    assert env.gen.calls == ["The hardware warranty is 48 months."]
    leaves = [n for n in env.store.rows["F1"] if n.metadata["node_role"] == "leaf"]
    # unchanged leaves keep their id, prefix and vector; the edited one is new
    kept = [n for n in leaves if n.metadata["chunk_hash"] in ids_before]
    assert len(kept) == 2 and all(ids_before[n.metadata["chunk_hash"]] == n.node_id for n in kept)
    assert all(n.metadata["context_prefix"].startswith("ctx:") for n in leaves)
    assert len(env.store.rows["F1"]) == 6  # 3 leaves + 3 parents, nothing duplicated


async def test_embedding_model_change_reembeds_but_keeps_prefixes(env):
    await _index(env, _doc())
    env.embed.model_name = "emb-2"
    again = await _index(env, _doc())
    assert (again["llm_calls"], again["embedded"]) == (0, 3)


async def test_prompt_change_regenerates_prefixes(env):
    await _index(env, _doc())
    env.gen.cache_key = "llm|prompt=2"
    again = await _index(env, _doc())
    assert (again["llm_calls"], again["embedded"]) == (3, 3)


async def test_failed_prefix_is_retried_next_time(env):
    async def _fail_one(chunks, **kw):
        env.gen.calls.extend(chunks)
        return [None if "warranty" in c else "ok" for c in chunks]

    env.gen.generate_prefixes = lambda chunks, full_document, file_name, **kw: _fail_one(chunks)
    await _index(env, _doc())
    warranty = next(n for n in env.store.rows["F1"] if "warranty" in n.text and n.metadata["node_role"] == "leaf")
    assert "ctx_key" not in warranty.metadata and "context_prefix" not in warranty.metadata
    again = await _index(env, _doc())
    assert again["llm_calls"] == 1


async def test_stored_text_is_original_and_search_text_has_breadcrumb(env):
    await _index(env, _doc())
    leaf = next(n for n in env.store.rows["F1"] if "warranty" in n.text and n.metadata["node_role"] == "leaf")
    assert leaf.text == "The hardware warranty is 36 months."
    assert env.store.search_texts[leaf.node_id] == (
        "ctx:The hardware\nAurora › 5. Warranty\nThe hardware warranty is 36 months.")
    parent = next(n for n in env.store.rows["F1"] if n.metadata["node_role"] == "parent"
                  and leaf.node_id in n.metadata["children_node_ids"])
    assert parent.node_id == parent_node_id("F1", [leaf.metadata["chunk_hash"]])
    assert leaf.node_id == leaf_node_id("F1", chunk_hash(leaf.text, "Aurora › 5. Warranty"))


def test_embedding_key_includes_passage_prefix():
    plain = SimpleNamespace(model_name="intfloat/multilingual-e5-large")
    wrapped = SimpleNamespace(model_name="intfloat/multilingual-e5-large", _inner=plain,
                              _passage_prefix="passage: ")
    assert embedding_key(plain) != embedding_key(wrapped)
    assert "passage=passage: " in embedding_key(wrapped)


async def test_prefix_scope_is_outline_plus_own_section(env):
    await _index(env, _doc())
    kw = env.gen.kwargs
    assert kw["outline"] == "- Aurora › 1. Overview\n- Aurora › 5. Warranty\n- Aurora › 6. Power"
    assert kw["contexts"][1] == "The hardware warranty is 36 months."  # its section only
    assert kw["language"] is None  # English: the prompt asks for the chunk's own language


async def test_segmented_mode_uses_the_configured_segmenter_and_stamps(env, monkeypatch):
    """Indexing writes the tsv with the same segmenter as Rebuild FTS and names it."""
    seen = {}

    def _replace(vector_store, file_id, nodes, search_texts, cfg, segment, segmenter_name=None):
        seen.update(cfg=cfg, segment=segment, name=segmenter_name)
        return len(nodes)

    from src.domain.rag import text_segmenter as ts
    seg = ts.BigramSegmenter()
    monkeypatch.setattr(hi, "replace_file_chunks", _replace)
    monkeypatch.setattr(hi.HierarchicalIndexer, "_text_search_config", staticmethod(lambda: "simple"))
    monkeypatch.setattr("src.domain.rag.text_segmenter.get_segmenter", lambda name=None: seg)
    await _index(env, _doc())
    assert seen["cfg"] == "simple" and seen["name"] == "bigram"
    assert seen["segment"] == seg.segment_batch


async def test_failed_prefixes_surface_as_file_warning(env):
    async def _fail_one(chunks, **kw):
        return [None if "warranty" in c else "ok" for c in chunks]

    env.gen.generate_prefixes = lambda chunks, full_document, file_name, **kw: _fail_one(chunks)
    stats = await _index(env, _doc())
    assert stats["prefix_failures"] == 1
    assert stats["warning"].startswith("1 of 3 chunks lack a context prefix")


async def test_parse_warning_and_prefix_warning_are_joined(env):
    async def _fail_all(chunks, **kw):
        return [None] * len(chunks)

    env.gen.generate_prefixes = lambda chunks, full_document, file_name, **kw: _fail_all(chunks)
    doc = _doc()
    doc.metadata["_parse_warning"] = "Pages 3 could not be parsed"
    stats = await _index(env, doc)
    assert stats["warning"].startswith("Pages 3 could not be parsed; 3 of 3 chunks")


async def test_no_warning_when_every_prefix_succeeds(env):
    stats = await _index(env, _doc())
    assert stats["prefix_failures"] == 0 and stats["warning"] is None


async def test_whitespace_only_prefix_is_not_cached(env):
    async def _blank(chunks, **kw):
        return ["  \x00 "] * len(chunks)

    env.gen.generate_prefixes = lambda chunks, full_document, file_name, **kw: _blank(chunks)
    stats = await _index(env, _doc())
    assert stats["prefix_failures"] == 3
    leaves = [n for n in env.store.rows["F1"] if n.metadata["node_role"] == "leaf"]
    assert all("ctx_key" not in n.metadata for n in leaves)


async def test_empty_prefix_cached_by_an_older_version_is_regenerated(env):
    await _index(env, _doc())
    # Simulate rows written by the old code: an empty prefix stamped with the live key.
    for n in env.store.rows["F1"]:
        if n.metadata["node_role"] == "leaf" and "warranty" in n.text:
            n.metadata["context_prefix"] = ""
    again = await _index(env, _doc())
    assert again["llm_calls"] == 1
    assert env.gen.calls == ["The hardware warranty is 36 months."]


def _table_doc(n_pieces=4, edit_last=False):
    chunks = [{"text": "Intro paragraph.", "headings": ["Plan", "Prices"]}]
    for k in range(n_pieces):
        row = f"| r{k} | {k * 10 + (1 if edit_last and k == n_pieces - 1 else 0)} |"
        chunks.append({"text": f"| item | price |\n| --- | --- |\n{row}",
                       "headings": ["Plan", "Prices"], "content_type": "table", "table_part": k})
    chunks.append({"text": "Closing note.", "headings": ["Plan", "Prices"]})
    return Document(text="full text", metadata={"file_id": "T1", "file_name": "Plan.xlsx",
                                                "_docling_chunks": chunks})


async def test_split_table_gets_one_prefix_call(env):
    stats = await _index(env, _table_doc(4))
    # intro + table head + closing = 3 calls, not 6
    assert stats["llm_calls"] == 3
    assert len(env.gen.calls) == 3
    leaves = [n for n in env.store.rows["T1"] if n.metadata["node_role"] == "leaf"]
    table = [n for n in leaves if n.metadata.get("content_type") == "table"]
    assert len(table) == 4
    assert len({n.metadata["context_prefix"] for n in table}) == 1
    assert all("table_part" not in n.metadata for n in leaves)  # transient marker
    assert all(n.metadata.get("ctx_key") for n in table)


async def test_table_continuation_reuses_cached_head_prefix(env):
    await _index(env, _table_doc(4))
    again = await _index(env, _table_doc(4, edit_last=True))
    assert again["llm_calls"] == 0  # only a continuation changed: it takes the head's prefix
    table = [n for n in env.store.rows["T1"]
             if n.metadata["node_role"] == "leaf" and n.metadata.get("content_type") == "table"]
    assert len({n.metadata["context_prefix"] for n in table}) == 1


async def test_failed_table_prefix_counts_every_piece(env):
    async def _fail_tables(chunks, **kw):
        return [None if "| item |" in c else "ok" for c in chunks]

    env.gen.generate_prefixes = lambda chunks, full_document, file_name, **kw: _fail_tables(chunks)
    stats = await _index(env, _table_doc(3))
    assert stats["prefix_failures"] == 3


def test_table_heads_need_same_headings():
    from src.domain.rag.hierarchical_indexer import _table_heads

    docs = [Document(text="a", metadata={"table_part": 0, "headings": ["A"]}),
            Document(text="b", metadata={"table_part": 1, "headings": ["A"]}),
            Document(text="c", metadata={"table_part": 1, "headings": ["B"]}),
            Document(text="d", metadata={}),
            Document(text="e", metadata={"table_part": 2, "headings": ["A"]})]
    assert _table_heads(docs) == [None, 0, None, None, None]


async def test_parents_are_sized_with_the_chunk_tokenizer(env):
    """Parents group leaves by real tokens: ~200-token CJK leaves, 1024 target -> 5 per parent."""
    from src.domain.rag.chunk_tokenizer import get_chunk_tokenizer

    tok = get_chunk_tokenizer(None)
    sent = "第{}條員工因公出差者，應於出發前三日內檢具出差申請單，經直屬主管核准後始得辦理。"
    chunks = [{"text": "".join(sent.format(i * 10 + k) for k in range(6)), "headings": ["Rules"]}
              for i in range(10)]
    sizes = {len(tok.encode(c["text"])) for c in chunks}
    assert max(sizes) <= 205
    doc = Document(text="full", metadata={"file_id": "P1", "file_name": "Rules.docx",
                                          "_docling_chunks": chunks})
    stats = await _index(env, doc)
    assert stats["parents"] == 2
