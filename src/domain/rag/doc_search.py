"""Search across a person's folders, with results as passages or as documents.

One search over several folders (all of the caller's, or the ``folders`` they name),
optionally limited by a document filter (src/domain/rag/doc_filter.py):

1. **Plan** (:func:`plan_search`, one short transaction). The caller's folders are
   read with the owner scope always applied; a named folder that is not theirs is
   "not found", exactly like a folder that does not exist. A filter is resolved to
   file ids per folder, and folders without a matching document drop out. Folders
   that cannot be searched are listed in ``skipped_folders`` with the reason:
   ``nothing_indexed``; ``embedding_model`` (some of its files were indexed with
   another embedding model or dimension than the configured one, so its vectors and
   the query's are not comparable); ``folder_limit`` (more searchable folders than
   ``rag.retrieval.max_search_folders``; the most recently updated ones are searched).
2. **Candidates per folder**, in parallel (at most :data:`FANOUT_CONCURRENCY` folders
   at a time, each inside the server-wide query gate), with the folder's own
   searcher: its dense table and its own BM25 segmenter. The query is embedded once
   and segmented once per segmenter. With N folders each folder gives a smaller pool
   (:func:`folder_depths`).
3. **One ranking over the pool.** The folders' candidate lists are fused by
   reciprocal rank (:func:`fuse_folders`), never by raw scores (cosine values from
   two tables are not comparable, ts_rank even less). With a reranker the top
   :data:`MAX_POOLED_CANDIDATES` of that order are reranked once: cross-encoder
   scores are query-passage scores and comparable across folders. Without one (or
   when the rerank call fails) the RRF order is the result.
4. **Passages** keep the top ``top_k``, auto-merged per folder as on one folder.
   **Documents** group every ranked passage by file: a document scores its best
   passage (MaxP), ties go to the one with more matching passages, and it carries
   its best :data:`DEFAULT_SNIPPETS` passages. Names, tags and metadata are read
   live from ``Files`` / ``Folders`` (:func:`describe_files`), so a rename shows at
   once, and a document whose file row is gone (or is not the caller's) is dropped.

Results are cached for :data:`~src.infrastructure.cache.cache_service.CacheTTL.QUERY_RESULT`
seconds under a key that covers the query, the folders and their filtered file ids,
the filter, ``group_by``, ``top_k`` and every retrieval setting; each searched folder
also gets a marker under its ``query:<folder id>:*`` namespace, which every index,
tag and metadata change of that folder already clears — an entry whose markers are
not all present is a miss (:class:`CrossFolderCache`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from src.domain.rag.doc_filter import ALL_FOLDERS, DocumentFilter, Owner, build_where
from src.domain.rag.doc_meta import stored_metadata, stored_tags
from src.log import get_api_logger

logger = get_api_logger()

GROUP_PASSAGE = "passage"
GROUP_DOCUMENT = "document"
GROUPS = (GROUP_PASSAGE, GROUP_DOCUMENT)

SKIP_NOT_INDEXED = "nothing_indexed"
SKIP_EMBEDDING = "embedding_model"
SKIP_LIMIT = "folder_limit"
SKIP_ERROR = "error"
_SKIP_DETAIL = {
    SKIP_NOT_INDEXED: "nothing is indexed in this folder yet",
    SKIP_EMBEDDING: ("some files were indexed with another embedding model or dimension; "
                     "re-index the folder to include it"),
    SKIP_LIMIT: "more folders than rag.retrieval.max_search_folders; name the folders to search",
    SKIP_ERROR: "the search of this folder failed (see the server log)",
}

DEFAULT_MAX_FOLDERS = 20
DEFAULT_DOCUMENT_TOP_K = 10
MAX_TOP_K = 50
DEFAULT_SNIPPETS = 3
MAX_SNIPPETS = 3
# Folders searched at once by one request (each also holds a slot of the
# server-wide query gate while it runs).
FANOUT_CONCURRENCY = 4
# Per-folder pool floor when several folders share one search (see folder_depths).
MIN_FOLDER_DENSE = 10
MIN_FOLDER_SPARSE = 4
# The pooled candidates the cross-encoder scores at most (~6 ms per pair).
MAX_POOLED_CANDIDATES = 150
RRF_K = 60


# ------------------------------------------------------------------ plan


@dataclass(frozen=True)
class FolderRef:
    id: int
    name: str
    vector_table_uuid: str
    owner: Optional[str]          # the folder's owner key (Folders.user_token)

    def ref(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name}


@dataclass(frozen=True)
class FolderTarget:
    folder: FolderRef
    file_ids: Optional[Tuple[str, ...]] = None   # a filter's matches; None = the whole folder


@dataclass
class SearchPlan:
    targets: List[FolderTarget] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)


class FolderNotFound(LookupError):
    """A named folder is not one of the caller's (or does not exist): reported the same."""

    def __init__(self, name):
        super().__init__(f"Folder not found: {name}")
        self.name = name


def _skip(folder: FolderRef, reason: str) -> Dict[str, Any]:
    return {**folder.ref(), "reason": reason, "detail": _SKIP_DETAIL[reason]}


def owned_folders(conn, owner: Owner, wanted: Optional[Sequence[Any]] = None) -> List[FolderRef]:
    """The caller's folders, most recently updated first; with ``wanted`` (names or
    ids) only those, and FolderNotFound for one that is not the caller's."""
    from sqlalchemy import text

    params: Dict[str, Any] = {}
    where = "TRUE"
    if isinstance(owner, str) and owner:
        where = "d.user_token = :owner"
        params["owner"] = owner
    elif owner is not ALL_FOLDERS:
        raise ValueError("an owner key (or ALL_FOLDERS) is required")
    rows = conn.execute(text(
        'SELECT d.id, d.name, d.vector_table_uuid, d.user_token FROM "Folders" d '
        f"WHERE {where} ORDER BY d.updated_at DESC NULLS LAST, d.id DESC"), params).fetchall()
    folders = [FolderRef(int(r[0]), r[1], str(r[2]), r[3]) for r in rows]
    if wanted is None:
        return folders
    by_id = {f.id: f for f in folders}
    by_name: Dict[str, FolderRef] = {}
    for f in folders:
        by_name.setdefault(f.name, f)
    out: List[FolderRef] = []
    for w in wanted:
        f = (by_id.get(int(w)) if isinstance(w, int) and not isinstance(w, bool)
             else by_name.get(w) if isinstance(w, str) else None)
        if f is None:
            raise FolderNotFound(w)
        if f not in out:
            out.append(f)
    return out


def folder_embedding_state(conn, folder_ids: Sequence[int],
                           current: Tuple[Optional[str], Optional[int]]) -> Dict[int, Dict[str, int]]:
    """Per folder: {"indexed": files indexed, "other_embedding": those indexed with
    another embedding model or dimension than ``current``} (chunk_identity.reindex_reason)."""
    from sqlalchemy import text

    from src.domain.rag.chunk_identity import (
        CHUNK_PIPELINE_VERSION, REINDEX_EMBEDDING, reindex_reason)

    out = {int(fid): {"indexed": 0, "other_embedding": 0} for fid in folder_ids}
    if not folder_ids:
        return out
    rows = conn.execute(text(
        'SELECT fi.folder_id, fi.embedding_model, fi.embedding_dimension, count(*) '
        'FROM "FileIndices" fi WHERE fi.status = \'indexed\' AND fi.folder_id = ANY(:ids) '
        "GROUP BY 1, 2, 3"), {"ids": [int(x) for x in folder_ids]}).fetchall()
    for fid, model, dim, n in rows:
        st = out.setdefault(int(fid), {"indexed": 0, "other_embedding": 0})
        st["indexed"] += int(n)
        # The current pipeline is passed: only the embedding matters for comparability.
        if reindex_reason("indexed", CHUNK_PIPELINE_VERSION, model, dim, current) == REINDEX_EMBEDDING:
            st["other_embedding"] += int(n)
    return out


def filtered_file_ids(conn, flt: Optional[DocumentFilter], *, owner: Owner,
                      folder_ids: Sequence[int]) -> Optional[Dict[int, Tuple[str, ...]]]:
    """{folder id: ids of its documents matching ``flt``} within the owner scope; None
    when there is no filter (every document)."""
    if flt is None or flt.is_empty():
        return None
    from sqlalchemy import text

    where, params = build_where(flt, owner=owner)
    rows = conn.execute(text(
        f'SELECT f.id, f.folder_id FROM "Files" f WHERE {where} AND f.folder_id = ANY(:_fids)'),
        {**params, "_fids": [int(x) for x in folder_ids]}).fetchall()
    out: Dict[int, List[str]] = {}
    for fid, folder_id in rows:
        out.setdefault(int(folder_id), []).append(str(fid))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def plan_search(conn, *, owner: Owner, folders: Optional[Sequence[Any]] = None,
                flt: Optional[DocumentFilter] = None, max_folders: int = DEFAULT_MAX_FOLDERS,
                current_embedding: Tuple[Optional[str], Optional[int]] = (None, None)
                ) -> SearchPlan:
    """Which of the caller's folders to search (with their filtered file ids) and which
    to skip, and why. See the module docstring."""
    refs = owned_folders(conn, owner, folders)
    plan = SearchPlan()
    ids = filtered_file_ids(conn, flt, owner=owner, folder_ids=[f.id for f in refs])
    if ids is not None:
        # A folder with no matching document has nothing to search: not a skip.
        refs = [f for f in refs if ids.get(f.id)]
    state = folder_embedding_state(conn, [f.id for f in refs], current_embedding)
    searchable: List[FolderTarget] = []
    for f in refs:
        st = state.get(f.id, {"indexed": 0, "other_embedding": 0})
        if st["indexed"] == 0:
            plan.skipped.append(_skip(f, SKIP_NOT_INDEXED))
        elif st["other_embedding"]:
            plan.skipped.append(_skip(f, SKIP_EMBEDDING))
        else:
            searchable.append(FolderTarget(f, ids.get(f.id) if ids is not None else None))
    cap = max(1, int(max_folders))
    plan.targets = searchable[:cap]
    plan.skipped.extend(_skip(t.folder, SKIP_LIMIT) for t in searchable[cap:])
    return plan


# ------------------------------------------------------------- ranking


def folder_depths(settings, n_folders: int) -> Tuple[int, int]:
    """(dense, BM25) pool depth per folder when ``n_folders`` share one search: one
    folder gets the usual ``per_list`` depths; N folders split twice that budget,
    never below MIN_FOLDER_DENSE / MIN_FOLDER_SPARSE each."""
    from src.domain.rag.hybrid_search import per_list_depths

    dense_k, sparse_k = per_list_depths(settings)
    if n_folders <= 1:
        return dense_k, sparse_k
    return (max(MIN_FOLDER_DENSE, math.ceil(2 * dense_k / n_folders)),
            max(MIN_FOLDER_SPARSE, math.ceil(2 * sparse_k / n_folders)))


def node_id_of(n) -> str:
    return n.node.node_id


def fuse_folders(lists: Sequence[Sequence[Any]], rrf_k: int = RRF_K) -> List[Any]:
    """One order over several folders' ranked candidate lists: reciprocal rank fusion
    (1 / (rrf_k + rank) per list), so only each folder's own order counts. Ties keep
    the order of the lists. The nodes get their RRF score."""
    scores: Dict[str, float] = {}
    first: Dict[str, Tuple[int, int]] = {}
    nodes: Dict[str, Any] = {}
    for li, lst in enumerate(lists):
        for rank, n in enumerate(lst):
            nid = node_id_of(n)
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (rrf_k + rank + 1)
            nodes.setdefault(nid, n)
            first.setdefault(nid, (rank, li))
    ranked = sorted(nodes, key=lambda nid: (-scores[nid], first[nid]))
    out = []
    for nid in ranked:
        n = nodes[nid]
        n.score = scores[nid]
        out.append(n)
    return out


@dataclass
class DocumentHit:
    file_id: str
    folder_id: int
    score: float
    hits: int
    snippets: List[Dict[str, Any]]
    best_rank: int


def _snippet(n) -> Dict[str, Any]:
    md = n.node.metadata or {}
    return {"text": n.node.text, "score": float(n.score) if n.score is not None else None,
            "page": md.get("page_no"), "headings": md.get("headings"),
            "content_type": md.get("content_type"), "node_id": n.node.node_id}


def group_documents(nodes: Sequence[Any], folder_of: Dict[str, int], *, top_k: int,
                    snippets: int = DEFAULT_SNIPPETS) -> List[DocumentHit]:
    """Ranked passages → ranked documents. A document scores its best passage (MaxP);
    a tie goes to the document with more matching passages, then to the one whose
    best passage ranked first. Each keeps its best ``snippets`` passages. Passages
    without a file id are left out."""
    docs: Dict[str, DocumentHit] = {}
    keep = max(1, min(int(snippets), MAX_SNIPPETS))
    for rank, n in enumerate(nodes):
        fid = (n.node.metadata or {}).get("file_id")
        if not fid:
            continue
        fid = str(fid)
        score = float(n.score) if n.score is not None else 0.0
        d = docs.get(fid)
        if d is None:
            docs[fid] = DocumentHit(file_id=fid, folder_id=folder_of.get(node_id_of(n), 0),
                                    score=score, hits=1, snippets=[_snippet(n)], best_rank=rank)
            continue
        d.hits += 1
        if score > d.score:
            d.score, d.best_rank = score, rank
        d.snippets.append(_snippet(n))
    for d in docs.values():
        d.snippets.sort(key=lambda s: -(s["score"] or 0.0))
        del d.snippets[keep:]
    ranked = sorted(docs.values(), key=lambda d: (-d.score, -d.hits, d.best_rank))
    return ranked[:max(1, int(top_k))]


def describe_files(conn, file_ids: Sequence[str], *, owner: Owner) -> Dict[str, Dict[str, Any]]:
    """{file id: name, mime, tags, metadata, folder {id, name}} read live, only for the
    caller's files (an id that is not a UUID, gone or not theirs is absent)."""
    from sqlalchemy import text

    good = []
    for fid in file_ids:
        try:
            good.append(str(uuid.UUID(str(fid))))
        except ValueError:
            continue
    if not good:
        return {}
    where, params = build_where(None, owner=owner)
    rows = conn.execute(text(
        "SELECT f.id, f.file_name, f.mime_type, f.tags, f.metadata, d.id AS folder_id, "
        'd.name AS folder_name FROM "Files" f JOIN "Folders" d ON d.id = f.folder_id '
        f"WHERE f.id = ANY(CAST(:_ids AS uuid[])) AND {where}"),
        {**params, "_ids": good}).mappings().fetchall()
    return {str(r["id"]): {
        "name": r["file_name"], "mime": r["mime_type"], "tags": stored_tags(r["tags"]),
        "metadata": stored_metadata(r["metadata"]),
        "folder": {"id": r["folder_id"], "name": r["folder_name"]}} for r in rows}


def documents_payload(hits: Sequence[DocumentHit], described: Dict[str, Dict[str, Any]]
                      ) -> List[Dict[str, Any]]:
    """The documents as returned: only those :func:`describe_files` found."""
    out = []
    for h in hits:
        info = described.get(h.file_id)
        if info is None:
            continue
        out.append({"file_id": h.file_id, "name": info["name"], "folder": info["folder"],
                    "mime": info["mime"], "tags": info["tags"], "metadata": info["metadata"],
                    "score": h.score, "hits": h.hits, "snippets": list(h.snippets)})
    return out


# ------------------------------------------------------------------- cache


class CrossFolderCache:
    """Result cache of searches across folders (see the module docstring)."""

    VERSION = "xf1"

    def __init__(self, cache, ttl: int):
        self._cache, self._ttl = cache, ttl

    @classmethod
    def fingerprint(cls, query: str, settings, *, targets: Sequence[FolderTarget], group_by: str,
                    top_k: int, snippets: int, reranker, merge_threshold,
                    filter_key: Optional[str]) -> str:
        """Every input that changes the answer: the retrieval settings, reranker,
        segmenter and merge threshold (hybrid_search.cache_fingerprint), the folders
        with their filtered file ids, the filter itself, group_by, top_k, snippets."""
        from src.domain.rag.hybrid_search import cache_fingerprint

        extra = json.dumps({
            "v": cls.VERSION, "group_by": group_by, "top_k": int(top_k), "snippets": int(snippets),
            "folders": [[t.folder.id, t.folder.vector_table_uuid,
                         list(t.file_ids) if t.file_ids is not None else None] for t in targets],
            "filter": filter_key or ""}, sort_keys=True, ensure_ascii=False)
        return cache_fingerprint(query, settings, reranker=reranker,
                                 merge_threshold=merge_threshold, filter_key=extra)

    @staticmethod
    def key(fp: str) -> str:
        return f"search:{fp}"

    @staticmethod
    def marker(folder_id: int, fp: str) -> str:
        # Inside the folder's query:<id>:* namespace, which its invalidations clear.
        return f"query:{int(folder_id)}:xf:{fp}"

    def get(self, fp: str, folder_ids: Sequence[int]):
        value = self._cache.get(self.key(fp))
        if value is None:
            return None
        if not all(self._cache.get(self.marker(fid, fp)) is not None for fid in folder_ids):
            self._cache.delete(self.key(fp))
            return None
        return value

    def put(self, fp: str, folder_ids: Sequence[int], value) -> None:
        for fid in folder_ids:
            self._cache.set(self.marker(fid, fp), 1, ttl=self._ttl)
        self._cache.set(self.key(fp), value, ttl=self._ttl)


# ------------------------------------------------------------------ search


@dataclass
class RankedPool:
    """The outcome of one search across folders, before names are attached."""

    nodes: List[Any]                       # ranked passages (leaves), best first
    folder_of: Dict[str, int]              # node id -> folder id
    score_type: str                        # rerank | rrf
    errors: List[int] = field(default_factory=list)       # folders whose search failed
    not_indexed: List[int] = field(default_factory=list)  # folders with no table yet
    rerank_failed: bool = False
    timings: Dict[str, float] = field(default_factory=dict)


async def rank_across(query: str, targets: Sequence[FolderTarget], *, settings,
                      searcher_for: Callable[[FolderRef], Any], reranker=None,
                      gate: Optional[asyncio.Semaphore] = None, keep: int,
                      rerank_fn: Optional[Callable[..., Awaitable[Any]]] = None) -> RankedPool:
    """Candidates from every target folder, pooled and ranked once (module docstring,
    steps 2 and 3). ``keep``: how many ranked passages to return at most (with a
    reranker, how many the rerank keeps). ``gate``: the server-wide query semaphore."""
    from dataclasses import replace

    from src.domain.rag import hybrid_search as hs

    if rerank_fn is None:
        from src.domain.rag.rerank_policy import rerank as rerank_fn
    timings: Dict[str, float] = {}
    tok = hs._STAGE_TIMINGS.set(timings)
    try:
        searchers = [searcher_for(t.folder) for t in targets]
        if not searchers:
            return RankedPool(nodes=[], folder_of={}, score_type="rerank" if reranker else "rrf",
                              timings=timings)
        dense_k, sparse_k = folder_depths(settings, len(targets))
        with hs.shared_query_work():
            embedding = await searchers[0].embed_query(query)
            fanout = asyncio.Semaphore(FANOUT_CONCURRENCY)

            async def one(i: int):
                t = targets[i]
                s = replace(settings, file_ids=t.file_ids)
                async with fanout:
                    async with (gate if gate is not None else contextlib.nullcontext()):
                        try:
                            return await searchers[i].folder_pool(
                                query, s, dense_k=dense_k, sparse_k=sparse_k,
                                query_embedding=embedding)
                        except Exception as e:  # noqa: BLE001 - one folder must not fail all
                            logger.warning(f"search of folder {t.folder.id} failed: "
                                           f"{type(e).__name__}: {e}")
                            return e

            pools = await asyncio.gather(*(one(i) for i in range(len(targets))))
        out = RankedPool(nodes=[], folder_of={}, score_type="rrf", timings=timings)
        lists: List[List[Any]] = []
        for t, pool in zip(targets, pools):
            if isinstance(pool, BaseException):
                out.errors.append(t.folder.id)
                continue
            if pool is None:
                out.not_indexed.append(t.folder.id)
                continue
            for n in pool:
                out.folder_of.setdefault(node_id_of(n), t.folder.id)
            lists.append(list(pool))
        if out.errors and len(out.errors) == len(targets):
            raise next(p for p in pools if isinstance(p, BaseException))
        t0 = time.perf_counter()
        fused = fuse_folders(lists)
        hs.record_stage("fuse", t0)
        if reranker is not None and fused:
            candidates = fused[:MAX_POOLED_CANDIDATES]
            t0 = time.perf_counter()
            async with (gate if gate is not None else contextlib.nullcontext()):
                rr = await rerank_fn(reranker, query, list(candidates), max(1, int(keep)))
            hs.record_stage("rerank", t0)
            if not rr.failed:
                out.nodes, out.score_type = list(rr.nodes), "rerank"
                return out
            out.rerank_failed = True
            # rerank_policy cut the candidates in retrieval order; restore their RRF scores.
            fused = fuse_folders(lists)
        out.nodes = fused[:max(1, int(keep))]
        return out
    finally:
        hs._STAGE_TIMINGS.reset(tok)


async def merge_passages(nodes: Sequence[Any], folder_of: Dict[str, int],
                         stores: Dict[int, Any], merge_threshold) -> List[Any]:
    """Auto-merge ranked leaves per folder (each folder's parents live in its own
    table) and arrange them in rank order (merge_plan.arrange)."""
    from src.domain.rag.merge_plan import arrange, merging_enabled, plan_merges

    if not merging_enabled(merge_threshold):
        return list(nodes)
    per: Dict[int, List[Any]] = {}
    for n in nodes:
        per.setdefault(folder_of.get(node_id_of(n), 0), []).append(n)
    groups: Dict[str, Any] = {}
    for fid, leaves in per.items():
        store = stores.get(fid)
        if store is None:
            continue
        groups.update(await plan_merges(store, leaves, merge_threshold))
    return arrange(list(nodes), groups)


def item_folder(item, folder_of: Dict[str, int]) -> int:
    """The folder of a ranked leaf or of a merged parent (via its best hit)."""
    best = getattr(item, "best_hit", None)
    n = best if best is not None else item
    return folder_of.get(node_id_of(n), 0)


@dataclass
class SearchRequest:
    query: str
    folders: Optional[List[Any]] = None          # names or ids; None = all the caller's
    filters: Optional[DocumentFilter] = None
    group_by: str = GROUP_PASSAGE
    top_k: Optional[int] = None                  # None = passages: rag.retrieval.default_top_k; documents: 10
    snippets: int = DEFAULT_SNIPPETS
    auto_merge: Optional[bool] = None            # passages; None = rag.retrieval.auto_merging.enabled


@dataclass
class SearchDeps:
    """What a search across folders needs from the running service (the adapter wires
    them; tests pass fakes)."""

    connect: Callable[[], Any]                          # sync DB connection (context manager)
    searcher_for: Callable[[FolderRef], Any]            # the folder's HybridSearcher
    query_one: Callable[[FolderRef, int], Awaitable[Any]]  # one folder, passages: the existing path
    settings: Any                                       # SearchSettings
    reranker: Any = None
    gate: Optional[asyncio.Semaphore] = None
    cache: Optional[CrossFolderCache] = None
    merge_threshold: Any = None
    max_folders: int = DEFAULT_MAX_FOLDERS
    current_embedding: Tuple[Optional[str], Optional[int]] = (None, None)


def _passage_payload(item, folder_id: int) -> Dict[str, Any]:
    """A ranked leaf or merged parent as a JSON passage; its folder's name is attached
    when the response is shaped (live, also for a cached result)."""
    from src.domain.rag.query_engine import _to_result

    data = _to_result(item).model_dump(mode="json")
    data["folder"] = {"id": folder_id}
    return data


async def search_across(req: SearchRequest, *, owner: Owner, deps: SearchDeps) -> Dict[str, Any]:
    """POST /api/search (and, in 1.6, MCP folio_find): plan, rank, shape. Returns the
    response ``data``; raises FolderNotFound for a named folder that is not the caller's."""
    started = time.perf_counter()
    group_by = req.group_by if req.group_by in GROUPS else GROUP_PASSAGE
    default_k = (DEFAULT_DOCUMENT_TOP_K if group_by == GROUP_DOCUMENT
                 else int(getattr(deps.settings, "top_k", 5) or 5))
    top_k = max(1, min(int(req.top_k or default_k), MAX_TOP_K))
    snippets = max(1, min(int(req.snippets or DEFAULT_SNIPPETS), MAX_SNIPPETS))

    def _plan():
        with deps.connect() as conn:
            return plan_search(conn, owner=owner, folders=req.folders, flt=req.filters,
                               max_folders=deps.max_folders,
                               current_embedding=deps.current_embedding)

    plan = await asyncio.to_thread(_plan)
    refs = {t.folder.id: t.folder for t in plan.targets}
    data: Dict[str, Any] = {
        "query": req.query, "group_by": group_by, "top_k": top_k,
        "folders": [t.folder.ref() for t in plan.targets],
        "skipped_folders": list(plan.skipped),
        "score_type": "rerank" if deps.reranker is not None else "rrf",
    }

    def _finish(results_key: str, items: List[Dict[str, Any]], timings=None) -> Dict[str, Any]:
        data[results_key] = items
        data["total_results"] = len(items)
        data["retrieval_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if timings:
            data["timings_ms"] = dict(timings)
        return data

    key = "documents" if group_by == GROUP_DOCUMENT else "results"
    if not plan.targets:
        return _finish(key, [])

    # One folder, passages: exactly the one-folder search (/api/rag/query), cache included.
    if group_by == GROUP_PASSAGE and len(plan.targets) == 1:
        folder = plan.targets[0].folder
        one = await deps.query_one(folder, top_k)
        items = []
        for r in one.results:
            d = r.model_dump(mode="json")
            d["metadata"]["folder_name"] = folder.name
            d["folder"] = folder.ref()
            items.append(d)
        if getattr(one, "not_indexed", False):
            data["folders"] = []
            data["skipped_folders"].append(_skip(folder, SKIP_NOT_INDEXED))
        data["score_type"] = getattr(one, "score_type", None) or data["score_type"]
        return _finish(key, items)

    fp = None
    folder_ids = [t.folder.id for t in plan.targets]
    if deps.cache is not None:
        fp = CrossFolderCache.fingerprint(
            req.query, deps.settings, targets=plan.targets, group_by=group_by, top_k=top_k,
            snippets=snippets, reranker=deps.reranker,
            merge_threshold=deps.merge_threshold if group_by == GROUP_PASSAGE else None,
            filter_key=req.filters.fingerprint() if req.filters is not None else None)
        hit = deps.cache.get(fp, folder_ids)
        if hit is not None:
            return await _shape(hit, req, data, refs, owner, deps, _finish, key, cached=True)

    keep = top_k if group_by == GROUP_PASSAGE else MAX_POOLED_CANDIDATES
    pool = await rank_across(req.query, plan.targets, settings=deps.settings,
                             searcher_for=deps.searcher_for, reranker=deps.reranker,
                             gate=deps.gate, keep=keep)
    if group_by == GROUP_PASSAGE:
        stores = {t.folder.id: deps.searcher_for(t.folder).vector_store for t in plan.targets}
        ranked: Any = await merge_passages(pool.nodes, pool.folder_of, stores, deps.merge_threshold)
        value = {"items": [_passage_payload(it, item_folder(it, pool.folder_of)) for it in ranked]}
    else:
        value = {"hits": group_documents(pool.nodes, pool.folder_of, top_k=top_k, snippets=snippets)}
    value.update(score_type=pool.score_type, errors=list(pool.errors),
                 not_indexed=list(pool.not_indexed), timings=dict(pool.timings))
    if (deps.cache is not None and fp is not None and not pool.rerank_failed
            and not pool.errors and not pool.not_indexed):
        deps.cache.put(fp, folder_ids, value)
    return await _shape(value, req, data, refs, owner, deps, _finish, key, cached=False)


async def _shape(value, req, data, refs, owner, deps, finish, key, *, cached: bool):
    """Attach live names to a (possibly cached) ranked result."""
    data["score_type"] = value["score_type"]
    dropped = set(value["errors"]) | set(value["not_indexed"])
    for fid in value["errors"]:
        data["skipped_folders"].append(_skip(refs[fid], SKIP_ERROR))
    for fid in value["not_indexed"]:
        data["skipped_folders"].append(_skip(refs[fid], SKIP_NOT_INDEXED))
    data["folders"] = [f for f in data["folders"] if f["id"] not in dropped]
    data["cached"] = cached
    if "hits" in value:
        hits = value["hits"]

        def _describe():
            with deps.connect() as conn:
                return describe_files(conn, [h.file_id for h in hits], owner=owner)

        described = await asyncio.to_thread(_describe) if hits else {}
        return finish(key, documents_payload(hits, described), value["timings"])
    items = []
    for it in value["items"]:
        ref = refs.get(it["folder"]["id"])
        if ref is None:
            continue
        d = json.loads(json.dumps(it))     # a copy: the cached value stays as it was
        d["folder"] = ref.ref()
        d["metadata"]["folder_name"] = ref.name
        items.append(d)
    return finish(key, items, value["timings"])
