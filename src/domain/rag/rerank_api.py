"""Rerank HTTP API styles — request shape and response parsing per provider family.

The Reranker (reranker.py) owns templates, thresholds and fallback; this module only
knows how to talk to each API:

    score    vLLM            POST /v1/score   {model, text_1, text_2[]}   → data[{index, score}]
    cohere   Cohere          POST /v1/rerank  {model, query, documents[], top_n}
                                                                         → results[{index, relevance_score}]
    jina     Jina AI         POST /v1/rerank  {model, query, documents[], top_n, return_documents: false}
                                                                         → results[{index, relevance_score}]
    generic  OpenAI-compatible rerank servers (vLLM /v1/rerank, Infinity, gateways)
                             POST /v1/rerank  {model, query, documents[], top_n}
                                                                         → results|data [{index, relevance_score|score}]

``endpoint_path`` overrides the default path (e.g. "/v2/rerank" for Cohere v2, "/rerank"
for a server mounted at the root). Every style returns one score per input document in
input order; a response that leaves documents unscored raises instead of inventing
scores (the Reranker then falls back to the retrieval order).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.http_retry import join_endpoint

STYLES = ("score", "cohere", "jina", "generic")
DEFAULT_PATHS = {
    "score": "/v1/score",
    "cohere": "/v1/rerank",
    "jina": "/v1/rerank",
    "generic": "/v1/rerank",
}


class RerankApi:
    def __init__(self, style: str, base_url: str, model: str, endpoint_path: Optional[str] = None):
        if style not in STYLES:
            raise ValueError(f"Unknown rerank api style {style!r} (expected one of {STYLES})")
        self.style = style
        self.model = model
        self.url = join_endpoint(base_url, endpoint_path or DEFAULT_PATHS[style])
        self.models_url = join_endpoint(base_url, "/v1/models")

    def build_payload(self, query: str, documents: List[str]) -> Dict[str, Any]:
        if self.style == "score":
            return {"model": self.model, "text_1": query, "text_2": list(documents)}
        payload: Dict[str, Any] = {"model": self.model, "query": query,
                                   "documents": list(documents), "top_n": len(documents)}
        if self.style == "jina":
            payload["return_documents"] = False
        return payload

    def parse_scores(self, data: Any, n: int) -> List[float]:
        if self.style == "score":
            items = data.get("data") if isinstance(data, dict) else None
        elif isinstance(data, list):
            items = data  # bare list (TEI-style servers behind a generic path)
        elif isinstance(data, dict):
            items = data.get("results")
            if items is None:
                items = data.get("data")
        else:
            items = None
        if not isinstance(items, list):
            raise ValueError(f"{self.style} rerank response has no result list")

        scores: List[Optional[float]] = [None] * n
        for item in items:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(idx, int) or not 0 <= idx < n or score is None:
                raise ValueError(f"{self.style} rerank response has an invalid item: {item!r}")
            scores[idx] = float(score)
        missing = sum(1 for s in scores if s is None)
        if missing:
            raise ValueError(f"{self.style} rerank response left {missing}/{n} documents unscored")
        return scores  # type: ignore[return-value]
