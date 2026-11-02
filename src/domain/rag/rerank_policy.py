"""Rerank policy — prompt templates, score threshold and failure handling.

The HTTP call stays behind the reranker interface (``Reranker.ascore`` /
``arerank``, whatever API style the provider speaks); this module decides what
the retrieval path does with the scores:

- **Templates.** ``mixedbread-ai/mxbai-rerank-*-v2`` is a Qwen2 model trained on
  a chat-formatted prompt. vLLM's score API joins the query and document text
  directly, so without the chat wrapper scores bunch up near the top (unrelated
  chunks score 0.8+) and no threshold can separate them. When
  ``rag.rerank.auto_template`` is on (default) the official format is applied
  automatically for that model family — unless the configured templates are
  custom (then they are left alone) or already chat-formatted.
- **Top score.** The best score *before* ``score_threshold`` is kept: knowledge
  gap analytics use it to tell "weak match" from "no match".
- **Failure.** If scoring fails the candidates are cut to ``top_n`` (they were
  3 × top_k) in retrieval order, and the outcome is marked failed so callers
  can skip caching a degraded result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from src.log import get_api_logger

logger = get_api_logger()

# Official mxbai-rerank-v2 prompt (the Qwen chat format of the model card / vLLM's
# mxbai_rerank_v2.jinja). vLLM concatenates query and document with no separator,
# so the query part ends with a newline.
MXBAI_V2_QUERY_TEMPLATE = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "query: {text}\n"
)
MXBAI_V2_DOCUMENT_TEMPLATE = (
    "document: {text}\n"
    "You are a search relevance expert who evaluates how well documents match search "
    "queries. For each query-document pair, carefully analyze the semantic relationship "
    "between them, then provide your binary relevance judgment (0 for not relevant, "
    "1 for relevant).\n"
    "Relevance:<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# With the chat template, mxbai-v2 scores are the probability of "relevant", so
# 0.5 is the model's own decision boundary.
MXBAI_V2_RECOMMENDED_THRESHOLD = 0.5


def is_mxbai_v2(model: Optional[str]) -> bool:
    m = (model or "").lower()
    return "mxbai-rerank" in m and "v2" in m


def _is_plain_or_legacy(query_template: str, document_template: str) -> bool:
    """Templates that are the default pass-through or the old chat-less mxbai form."""
    q = (query_template or "{text}").strip()
    d = (document_template or "{text}").strip()
    if q == "{text}" and d == "{text}":
        return True
    return q == "query: {text}" and d.startswith("document: {text}")


def effective_templates(model: Optional[str], query_template: str, document_template: str,
                        auto: bool = True) -> Tuple[str, str]:
    """The (query, document) templates the reranker should actually send."""
    qt, dt = query_template or "{text}", document_template or "{text}"
    if not auto or not is_mxbai_v2(model) or "<|im_start|>" in qt + dt:
        return qt, dt
    if _is_plain_or_legacy(qt, dt):
        return MXBAI_V2_QUERY_TEMPLATE, MXBAI_V2_DOCUMENT_TEMPLATE
    logger.warning(
        f"Reranker {model} has custom templates without the Qwen chat format; leaving them "
        "as configured. Scores may not be calibrated (see rag.rerank.auto_template).")
    return qt, dt


def resolve_rerank_config(rerank_config):
    """A copy of the rerank config with the effective templates filled in."""
    if rerank_config is None:
        return None
    qt, dt = effective_templates(
        getattr(rerank_config, "model", None),
        getattr(rerank_config, "query_template", "{text}"),
        getattr(rerank_config, "document_template", "{text}"),
        auto=bool(getattr(rerank_config, "auto_template", True)),
    )
    if (qt, dt) == (getattr(rerank_config, "query_template", None),
                    getattr(rerank_config, "document_template", None)):
        return rerank_config
    logger.info(f"Reranker {rerank_config.model}: applying the official chat template")
    try:
        return rerank_config.model_copy(update={"query_template": qt, "document_template": dt})
    except AttributeError:  # plain objects in tests / older configs
        import copy

        out = copy.copy(rerank_config)
        out.query_template, out.document_template = qt, dt
        return out


@dataclass
class RerankOutcome:
    nodes: List[Any]
    top_score: Optional[float] = None   # best score before the threshold; None if unscored
    failed: bool = False


async def rerank(reranker, query: str, nodes: List[Any], top_n: int) -> RerankOutcome:
    """Score, sort, apply ``score_threshold``, keep ``top_n``.

    Uses the reranker's raw ``ascore`` when it has one (so the pre-threshold top
    score is known); otherwise its ``arerank``.
    """
    if not nodes:
        return RerankOutcome(nodes=[])
    score_fn = getattr(reranker, "ascore", None)
    try:
        if score_fn is None:
            ranked = list(await reranker.arerank(query, list(nodes), top_n=top_n))[:top_n]
            top = ranked[0].score if ranked else None
            return RerankOutcome(nodes=ranked, top_score=top)
        scores = list(await score_fn(query, [n.text for n in nodes]))
        if len(scores) != len(nodes):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(nodes)} texts")
    except Exception as e:  # noqa: BLE001 — degrade, never fail the query
        logger.warning(f"Reranking failed, keeping the top {top_n} in retrieval order: {e}")
        return RerankOutcome(nodes=list(nodes)[:top_n], failed=True)

    threshold = float(getattr(reranker, "score_threshold", 0.0) or 0.0)
    pairs = sorted(zip(nodes, scores), key=lambda p: p[1], reverse=True)
    kept = []
    for node, score in pairs:
        if score < threshold:
            break
        node.score = float(score)
        kept.append(node)
        if len(kept) >= top_n:
            break
    top = float(pairs[0][1]) if pairs else None
    logger.debug(f"Reranked {len(nodes)} → {len(kept)} (threshold={threshold}, top={top})")
    return RerankOutcome(nodes=kept, top_score=top)
