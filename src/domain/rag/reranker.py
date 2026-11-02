
"""Reranker — Cross-encoder based result reranking

Calls a rerank HTTP API (vLLM score, Cohere, Jina or a generic OpenAI-compatible
/rerank — see rerank_api.py) to rerank retrieval results for better precision.

The reranker takes query-chunk pairs and produces relevance scores,
which are more accurate than bi-encoder cosine similarity.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import httpx
from src.domain.rag.rerank_api import RerankApi
from src.log import get_api_logger
# Rerank on what the chunk is retrieved by: prefix + breadcrumb + original text.
from src.domain.rag.chunk_identity import node_retrieval_text
from src.utils.http_retry import RetryPolicy, arequest_with_retry, request_with_retry

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

logger = get_api_logger()


class Reranker:
    """Cross-encoder reranker over a rerank HTTP API (style chosen by rag.rerank.api)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        top_n: Optional[int] = None,
        score_threshold: float = 0.0,
        timeout: float = 10.0,
        query_template: str = "{text}",
        document_template: str = "{text}",
        api: str = "score",
        endpoint_path: Optional[str] = None,
        max_retries: int = 1,
    ):
        """Initialize reranker

        Args:
            base_url: vLLM/OpenAI-compatible API base URL (e.g., http://localhost:8787)
            model: Model name (e.g., BAAI/bge-reranker-v2-m3)
            api_key: API key (default: not-needed for local vLLM)
            top_n: Keep top N results after reranking (None = keep all)
            score_threshold: Minimum rerank score to keep (default 0.0)
            timeout: HTTP request timeout in seconds
            query_template: query wrapping template; {text} is the placeholder (for instruction-tuned rerankers)
            document_template: document wrapping template; {text} is the placeholder
            api: HTTP API style — score (vLLM) | cohere | jina | generic
            endpoint_path: override the style's default path (e.g. "/v2/rerank")
            max_retries: retries on 429 / 5xx / timeouts before falling back to raw order
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._top_n = top_n
        self._score_threshold = score_threshold
        self._timeout = timeout
        # Use str.replace rather than str.format: chunk text may contain {} braces, which format would choke on
        self._query_template = query_template or "{text}"
        self._document_template = document_template or "{text}"
        # async client is created lazily (must be built inside an event loop); the connection pool is reused across queries
        self._async_client: Optional[httpx.AsyncClient] = None
        self._api = RerankApi(api, self._base_url, model, endpoint_path)
        self._retry = RetryPolicy(max_retries=max_retries, backoff_base=0.25, backoff_max=2.0,
                                  retry_after_max=5.0)  # query path: keep retries short

        logger.info(
            f"Reranker initialized (api={api}, model={model}, base_url={base_url}, "
            f"top_n={top_n}, score_threshold={score_threshold})"
        )

    @classmethod
    def from_config(cls, rerank_config) -> Optional["Reranker"]:
        """Create Reranker from config

        Returns:
            Reranker instance, or None if not configured/enabled
        """
        if not rerank_config or not rerank_config.enabled:
            logger.info("Reranker is disabled")
            return None

        try:
            instance = cls(
                base_url=rerank_config.base_url,
                model=rerank_config.model,
                api_key=getattr(rerank_config, 'api_key', 'not-needed') or 'not-needed',
                top_n=getattr(rerank_config, 'top_n', None),
                score_threshold=getattr(rerank_config, 'score_threshold', 0.0),
                query_template=getattr(rerank_config, 'query_template', '{text}'),
                document_template=getattr(rerank_config, 'document_template', '{text}'),
                api=getattr(rerank_config, 'api', 'score') or 'score',
                endpoint_path=getattr(rerank_config, 'endpoint_path', None),
                timeout=float(getattr(rerank_config, 'timeout', 10.0) or 10.0),
                max_retries=int(getattr(rerank_config, 'max_retries', 1)),
            )
            # Run connection verification on a background thread — from_config is
            # called by the lazy RAGAdapter on the first request's event loop, so
            # running sync HTTP (3s timeout) inline would block the whole loop.
            # Verification only logs a warning, so fire-and-forget is fine.
            import threading
            threading.Thread(
                target=instance.verify_connection,
                daemon=True,
                name="rerank-verify",
            ).start()
            return instance
        except Exception as e:
            logger.error(f"Failed to create Reranker: {e}")
            return None

    def verify_connection(self) -> None:
        """Best-effort startup check that the rerank service serves this model name.

        If the model name does not match what vllm serves, /v1/score returns 404,
        and rerank() falls back to the raw order on any error (only a warning log) —
        effectively silently disabling reranking. This raises a loud error at startup
        without blocking it (the service may come up later).
        """
        if self._api.style in ("cohere", "jina"):
            return  # hosted rerank APIs do not list models; the console probe covers them
        try:
            resp = httpx.get(
                self._api.models_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=3.0,
            )
            resp.raise_for_status()
            served = [m.get("id") for m in resp.json().get("data", [])]
            if self._model not in served:
                logger.error(
                    f"Reranker model '{self._model}' is NOT served at "
                    f"{self._base_url} (served: {served}). {self._api.url} will 404 and "
                    f"rerank will silently fall back to raw retrieval order — "
                    f"fix config rag.rerank.model or the vLLM service."
                )
        except Exception as e:
            logger.warning(
                f"Reranker connectivity check failed ({self._base_url}): {e}. "
                f"If the service stays unreachable, rerank silently falls back "
                f"to raw retrieval order."
            )

    def rerank(
        self,
        query: str,
        nodes: List["NodeWithScore"],
        top_n: Optional[int] = None,
    ) -> List["NodeWithScore"]:
        """Rerank nodes using cross-encoder scores

        Args:
            query: Original query text
            nodes: Retrieved nodes to rerank
            top_n: Override default top_n (optional)

        Returns:
            Reranked list of NodeWithScore (sorted by rerank score, descending)
        """
        if not nodes:
            return nodes

        effective_top_n = top_n or self._top_n

        try:
            scores = self._get_scores(query, [node_retrieval_text(n) for n in nodes])

            scored_nodes = list(zip(nodes, scores))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)

            result = []
            for node, score in scored_nodes:
                if score < self._score_threshold:
                    continue
                node.score = score
                result.append(node)
                if effective_top_n and len(result) >= effective_top_n:
                    break

            logger.info(
                f"Reranked {len(nodes)} → {len(result)} results "
                f"(top_n={effective_top_n}, threshold={self._score_threshold})"
            )
            return result

        except Exception as e:
            # Keep only top_n: the candidates are 3 × top_k, and handing them all
            # back made a reranker outage triple the result count.
            logger.warning(f"Reranking failed, keeping the top {effective_top_n} in retrieval order: {e}")
            return nodes[:effective_top_n] if effective_top_n else nodes

    async def arerank(
        self,
        query: str,
        nodes: List["NodeWithScore"],
        top_n: Optional[int] = None,
    ) -> List["NodeWithScore"]:
        """Async version of rerank() — used on the query path so scoring HTTP does not occupy the event loop.

        Degradation behavior is identical to the sync version: any failure returns the original order (scores preserved).

        Args:
            query: original query string.
            nodes: retrieval results to rerank.
            top_n: override the default top_n (optional).

        Returns:
            Reranked NodeWithScore list (scores descending).
        """
        if not nodes:
            return nodes

        effective_top_n = top_n or self._top_n

        try:
            scores = await self._aget_scores(query, [node_retrieval_text(n) for n in nodes])

            scored_nodes = list(zip(nodes, scores))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)

            result = []
            for node, score in scored_nodes:
                if score < self._score_threshold:
                    continue
                node.score = score
                result.append(node)
                if effective_top_n and len(result) >= effective_top_n:
                    break

            logger.info(
                f"Reranked {len(nodes)} → {len(result)} results "
                f"(top_n={effective_top_n}, threshold={self._score_threshold})"
            )
            return result

        except Exception as e:
            # Keep only top_n: the candidates are 3 × top_k, and handing them all
            # back made a reranker outage triple the result count.
            logger.warning(f"Reranking failed, keeping the top {effective_top_n} in retrieval order: {e}")
            return nodes[:effective_top_n] if effective_top_n else nodes

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    async def ascore(self, query: str, texts: List[str]) -> List[float]:
        """Raw relevance scores, one per text, in input order. Raises on failure
        (the retrieval layer decides how to degrade; see rerank_policy)."""
        return await self._aget_scores(query, texts)

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}

    def _payload(self, query: str, texts: List[str]) -> dict:
        # Templates use str.replace (chunk text may contain {} braces that format would choke on)
        return self._api.build_payload(
            self._query_template.replace("{text}", query),
            [self._document_template.replace("{text}", t) for t in texts],
        )

    async def _aget_scores(self, query: str, texts: List[str]) -> List[float]:
        """Async scores via the configured API style (httpx.AsyncClient, pooled across queries).

        Retries 429 / 5xx / timeouts per rag.rerank.max_retries; after that every error
        propagates so arerank() falls back to the original order — it never returns
        all-zero scores (which the threshold would filter into an empty result).
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)

        response = await arequest_with_retry(
            self._async_client, "POST", self._api.url, policy=self._retry,
            json=self._payload(query, texts), headers=self._headers(),
        )
        response.raise_for_status()
        return self._api.parse_scores(response.json(), len(texts))

    def _get_scores(self, query: str, texts: List[str]) -> List[float]:
        """Sync scores via the configured API style (one score per text, input order).

        WARNING: timeout / connection errors must not be swallowed here into
        all-zero scores — all zeros get fully filtered by the upstream
        score_threshold, so the query returns an "empty result", which to the
        caller is a wrong answer meaning "no data found". Let the exception
        propagate so rerank()'s except Exception uniformly falls back to the raw
        retrieval order (scores preserved), which is the correct degradation.
        """
        with httpx.Client(timeout=self._timeout) as client:
            response = request_with_retry(
                client, "POST", self._api.url, policy=self._retry,
                json=self._payload(query, texts), headers=self._headers(),
            )
        response.raise_for_status()
        return self._api.parse_scores(response.json(), len(texts))
