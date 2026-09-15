
"""PrefixedEmbedding — wrapper that auto-prepends query/passage prefixes for e5-family models.

The e5 models (multilingual-e5-*) are trained with a prefix contract: query text
gets ``query: `` and document text gets ``passage: ``. Omitting the prefix noticeably
degrades retrieval quality. The bge family needs no prefix.

llama_index's BaseEmbedding already separates "query" from "document" into distinct
methods (_get_query_embedding vs _get_text_embedding), so we just intercept each side
and prepend the prefix; upstream (indexer / query engine) needs no changes. When both
prefixes are empty strings, rag_context skips this wrapper, so the bge path has zero
extra overhead.
"""

from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr


class PrefixedEmbedding(BaseEmbedding):
    """Wrap any BaseEmbedding, prepending a fixed prefix to queries and documents."""

    _inner: Any = PrivateAttr()
    _query_prefix: str = PrivateAttr()
    _passage_prefix: str = PrivateAttr()

    def __init__(self, inner: BaseEmbedding, query_prefix: str = "", passage_prefix: str = "", **kwargs):
        super().__init__(
            model_name=inner.model_name,
            embed_batch_size=inner.embed_batch_size,
            **kwargs,
        )
        self._inner = inner
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._inner._get_query_embedding(self._query_prefix + query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._inner._aget_query_embedding(self._query_prefix + query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._inner._get_text_embedding(self._passage_prefix + text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._inner._get_text_embeddings([self._passage_prefix + t for t in texts])

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._inner._aget_text_embedding(self._passage_prefix + text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._inner._aget_text_embeddings([self._passage_prefix + t for t in texts])