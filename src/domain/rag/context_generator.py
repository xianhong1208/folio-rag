
"""Contextual Retrieval — Context Generator

Generates a context prefix for each chunk, resolving the semantic ambiguity a chunk has once
detached from its source document.
Reference: Anthropic "Introducing Contextual Retrieval" (2024-09)

Usage:
    generator = ContextGenerator.from_config(rag_config)
    contextualized_texts = await generator.generate_batch(
        chunks=["The circumstances described in the preceding paragraph are not subject to this article.", ...],
        full_document="Full text of the travel policy...",
        file_name="travel_policy.docx",
    )

Performance:
    Uses a sync OpenAI client + asyncio.to_thread to call the LLM concurrently, throttled by an
    asyncio.Semaphore, maximizing vLLM continuous-batching efficiency. The sync client avoids the
    AsyncOpenAI event-loop issues seen under PyInstaller-packaged environments.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional, List

from src.log import get_api_logger

logger = get_api_logger()

# The LLM must output in the SAME language as the source document, keeping the
# context prefix embedding-aligned with the chunk it precedes; a hard-coded English
# prefix would drift in bge-m3 vector space and break cross-language retrieval.
# Few-shot examples and a hard language rule are needed because gemma-3-27b otherwise
# slips into English on Chinese transcripts.
_CONTEXT_PROMPT = """\
You are a document analysis assistant. Generate a short context line (1-2 sentences) \
describing where the given chunk sits in the document, so a downstream retriever can \
understand it in isolation.

═══ ABSOLUTE LANGUAGE RULE (NON-NEGOTIABLE) ═══
The output MUST be in the EXACT SAME language/script as the chunk text.
- Write the context in the chunk's own language (Chinese chunk → Chinese context).
- Japanese chunk → Japanese context.
- If chunk is in English, output in English.
Mixed-language output is FORBIDDEN. Output language is NOT optional.

═══ FEW-SHOT EXAMPLES (match the output language) ═══
Example 1 (Chinese chunk → Chinese output):
  Chunk: "This article requires the applicant for a building permit to attach the building plan..."
  Output: An extension of Article 3 of the construction act; this passage lists the documents a building-permit applicant must attach

Example 2 (English chunk → English output):
  Chunk: "Article 3 defines terms used in this Act, including 'construction work'..."
  Output: Article 3 of the Act, listing core terminology definitions

Example 3 (Chinese transcript chunk → Chinese output):
  Chunk: "Deputy director: this is our current SOP; we did not create a new procedure for this item"
  Output: The deputy director answers the chair, saying the case follows the current SOP with no new procedure

═══ OUTPUT CONSTRAINTS ═══
- Identify the speaker if obvious (transcripts) — e.g., chair / deputy director / representative
- Include the document name or topic title if relevant
- State WHAT the chunk discusses (do not paraphrase the chunk itself)
- Maximum {max_length} characters
- No quotation marks, no English "This chunk..." preamble
- If you cannot determine context, output a short topic in the same language as chunk

═══ NOW PROCESS ═══
Document name: {file_name}

<full_document>
{document_text}
</full_document>

<chunk>
{chunk_text}
</chunk>

Output the context line directly (remember: SAME LANGUAGE AS CHUNK):"""


def _detect_scripts(text: str) -> set:
    """Detect the set of major Unicode scripts present in text, deciding directly
    from the Unicode block (no hardcoded language). Returns e.g. {"Latin", "CJK"}.
    """
    scripts: set = set()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x0080:                              scripts.add("Latin")
        elif 0x00C0 <= cp < 0x0250:                  scripts.add("Latin")    # Latin Extended
        elif 0x0400 <= cp < 0x0500:                  scripts.add("Cyrillic")
        elif 0x0590 <= cp < 0x0600:                  scripts.add("Hebrew")
        elif 0x0600 <= cp < 0x0700:                  scripts.add("Arabic")
        elif 0x0900 <= cp < 0x0980:                  scripts.add("Devanagari")
        elif 0x0E00 <= cp < 0x0E80:                  scripts.add("Thai")
        elif 0x3040 <= cp < 0x30A0:                  scripts.add("Hiragana")
        elif 0x30A0 <= cp < 0x3100:                  scripts.add("Katakana")
        elif 0xAC00 <= cp < 0xD7B0:                  scripts.add("Hangul")
        elif 0x4E00 <= cp < 0xA000:                  scripts.add("CJK")
    return scripts


def _is_script_mismatch(chunk_text: str, candidate_prefix: str) -> bool:
    """Return True if the prefix's main script mismatches the chunk's (avoids vector drift).

    A non-Latin chunk (CJK / Hangul / Arabic / Cyrillic …) with an all-Latin prefix is
    a mismatch; the Latin tolerance allows incidental alphanumerics in any language.
    """
    chunk_scripts = _detect_scripts(chunk_text)
    prefix_scripts = _detect_scripts(candidate_prefix)

    # Exclude Latin: a pure-Latin chunk is fine with a Latin prefix
    chunk_main = chunk_scripts - {"Latin"}
    prefix_main = prefix_scripts - {"Latin"}

    return bool(chunk_main) and not prefix_main


def _safe_fallback_prefix(chunk_text: str, file_name: str) -> str:
    """Fallback prefix when the LLM fails: use file_name directly, adding no natural-
    language wrapper, so the prefix stays language-neutral (bge-m3 is fairly
    insensitive to naming-like noise). Returns "{file_name}\n\n{chunk_text}".
    """
    return f"{file_name}\n\n{chunk_text}"


class ContextGenerator:
    """Uses an LLM to generate a context prefix for each chunk.

    Uses a sync OpenAI client + asyncio.to_thread() for concurrent calls, avoiding AsyncOpenAI
    event-loop compatibility issues under PyInstaller-packaged environments.
    """

    def __init__(
        self,
        llm_config,
        model: str,
        max_context_length: int = 150,
        max_concurrent: int = 50,
        max_doc_chars: int = 60000,
        max_tokens: int = 1024,
        reasoning_effort: Optional[str] = None,
        max_calls_per_file: int = 0,
        max_tokens_per_day: int = 0,
        budget=None,
    ):
        self._llm_config = llm_config
        self._model = model
        self._max_context_length = max_context_length
        self._max_concurrent = max_concurrent
        self._max_doc_chars = max_doc_chars
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        # Cost guards (0 = unlimited): chunks past a cap get the fallback prefix, no LLM call.
        self._max_calls_per_file = max_calls_per_file
        self._max_tokens_per_day = max_tokens_per_day
        if budget is None:
            from src.domain.rag.usage_budget import BUDGET as budget
        self._budget = budget
        logger.info(
            f"ContextGenerator initialized (model={model}, max_length={max_context_length}, "
            f"max_concurrent={max_concurrent}, max_doc_chars={max_doc_chars}, "
            f"max_tokens={max_tokens}, reasoning_effort={reasoning_effort})"
        )

    @classmethod
    def from_config(cls, rag_config) -> Optional["ContextGenerator"]:
        """Build a ContextGenerator from the RAG config.

        Returns:
            ContextGenerator instance, or None if not configured/enabled
        """
        cr_config = getattr(rag_config, 'contextual_retrieval', None)
        if not cr_config or not cr_config.enabled:
            logger.info("Contextual Retrieval is disabled")
            return None

        from src.config.roles import resolve_llm
        llm_config = resolve_llm(rag_config, "contextual_llm")  # rag.contextual_llm over rag.llm
        if not llm_config:
            logger.warning("Contextual Retrieval enabled but no LLM config found")
            return None

        try:
            model = llm_config.azure_deployment or llm_config.model

            return cls(
                llm_config=llm_config,
                model=model,
                max_context_length=cr_config.max_context_length,
                max_concurrent=getattr(cr_config, 'max_concurrent', 50),
                max_doc_chars=getattr(cr_config, 'max_doc_chars', 60000),
                max_tokens=getattr(cr_config, 'max_tokens', 1024),
                reasoning_effort=getattr(cr_config, 'reasoning_effort', None),
                max_calls_per_file=int(getattr(cr_config, 'max_calls_per_file', 0) or 0),
                max_tokens_per_day=int(getattr(cr_config, 'max_tokens_per_day', 0) or 0),
            )
        except Exception as e:
            logger.error(f"Failed to create ContextGenerator: {e}")
            return None

    @staticmethod
    def _create_sync_client(llm_config):
        """Build a sync OpenAI client for the contextual-prefix role (see llm_client:
        every OpenAI-compatible provider honours base_url, timeout and max_retries)."""
        from src.domain.rag.llm_client import create_openai_client
        return create_openai_client(llm_config)

    def _call_llm_sync(self, prompt: str, client) -> str:
        """Call the LLM synchronously (run in a thread pool via asyncio.to_thread)."""
        kwargs = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": 0,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        # Read content first; if empty, fall back to reasoning_content (reasoning models)
        raw_content = msg.content
        if not raw_content:
            raw_content = getattr(msg, 'reasoning_content', None)
        return raw_content

    def _admit_tokens(self, prompt: str) -> bool:
        """Charge an estimated cost (prompt chars / 4 + max_tokens) to today's prefix budget."""
        limit = getattr(self, "_max_tokens_per_day", 0)
        if not limit:
            return True
        estimate = len(prompt) // 4 + self._max_tokens
        if self._budget.try_consume("prefix_tokens", estimate, limit):
            return True
        if not getattr(self, "_budget_warned", False):
            self._budget_warned = True
            logger.warning(
                f"Contextual-prefix daily token budget ({limit}) exhausted — "
                "remaining chunks get the file-name fallback prefix until UTC midnight")
        return False

    async def _generate_one(
        self,
        chunk_text: str,
        full_document: str,
        file_name: str,
        client,
        semaphore: asyncio.Semaphore,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Generate the context-prefixed text for a single chunk.

        When should_abort holds, return the original chunk_text and skip the LLM — after an abort
        the whole batch's results are discarded by the indexer (raises IndexingAbortedError), so
        the goal here is only to let gather finish quickly without wasting LLM calls.
        """
        try:
            if should_abort and should_abort():
                return chunk_text

            doc_text = full_document[:self._max_doc_chars]
            if len(full_document) > self._max_doc_chars:
                doc_text += "\n...(document too long, truncated)"

            prompt = _CONTEXT_PROMPT.format(
                max_length=self._max_context_length,
                file_name=file_name,
                document_text=doc_text,
                chunk_text=chunk_text,
            )
            if not self._admit_tokens(prompt):
                return _safe_fallback_prefix(chunk_text, file_name)

            async with semaphore:
                # A chunk queued behind the semaphore may have waited a while; re-check once it acquires the slot
                if should_abort and should_abort():
                    return chunk_text
                raw_content = await asyncio.to_thread(self._call_llm_sync, prompt, client)

            if not raw_content:
                logger.warning(
                    f"LLM returned empty content for chunk in '{file_name}',"
                    f" using safe fallback prefix"
                )
                return _safe_fallback_prefix(chunk_text, file_name)

            context = raw_content.strip()

            # A script-mismatched prefix+chunk drifts in bge-m3 space and tanks the
            # chunk's hit rate on same-language queries; prefer the fallback instead.
            if _is_script_mismatch(chunk_text, context):
                logger.warning(
                    f"LLM produced script-mismatched prefix for chunk in '{file_name}',"
                    f" discarding. context_preview={context[:60]!r}"
                )
                return _safe_fallback_prefix(chunk_text, file_name)

            if len(context) > self._max_context_length:
                context = context[:self._max_context_length]

            return f"{context}\n\n{chunk_text}"

        except Exception as e:
            logger.warning(f"Context generation failed for chunk in '{file_name}': {e}")
            return _safe_fallback_prefix(chunk_text, file_name)

    async def generate_batch(
        self,
        chunks: List[str],
        full_document: str,
        file_name: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> List[str]:
        """Generate contextual prefixes for a batch of chunks (concurrent LLM calls).

        Sync OpenAI client + asyncio.to_thread for concurrency; asyncio.Semaphore for throttling.

        Args:
            chunks: List of chunk texts to prefix.
            full_document: Full document content (the LLM needs it to reason about chunk context).
            file_name: File-name tag (for logging + fallback).
            on_progress: ``(done, total)`` progress callback; invoked every 10 completed chunks or
                on the last one (throttled, so job status can show progress). None = silent.
            should_abort: When it returns True, remaining chunks skip the LLM to finish quickly
                (cooperative cancellation for a file deleted mid-run). The caller guarantees this
                callable won't raise; abort adjudication and result discarding are the indexer's
                job, and this method returns normally.

        Returns:
            A list the same length as chunks; each element is the assembled `prefix\n\nchunk_text` string.
        """
        total = len(chunks)

        # One sync client per file; must be closed when done or a long-running server
        # leaks sockets/fds linearly with file count.
        client = self._create_sync_client(self._llm_config)
        semaphore = asyncio.Semaphore(self._max_concurrent)

        # Progress uses a completion count, not the chunk index, since gather's
        # completion order differs from submission order. Incremented only after await
        # (single-threaded within the event loop), so it needs no lock.
        done_count = 0

        def _on_one_done() -> None:
            nonlocal done_count
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                logger.info(f"Context generation progress: {done_count}/{total} chunks")
                if on_progress is not None:
                    try:
                        on_progress(done_count, total)
                    except Exception as e:
                        logger.debug(f"context-gen progress callback failed (ignored): {e}")

        # Abort watcher: on file deletion, also close the client immediately so vLLM
        # aborts the in-flight generations (client disconnect) rather than only
        # skipping subsequent calls. Covers the multi-file-job path (per-file abort
        # flag); a single-file job already breaks via task.cancel().
        _abort_stop = asyncio.Event()

        async def _abort_watcher():
            while not _abort_stop.is_set():
                if should_abort and should_abort():
                    logger.info(
                        f"Abort detected for '{file_name}' — closing LLM client "
                        f"to cut in-flight generations"
                    )
                    try:
                        client.close()
                    except Exception:
                        pass
                    return
                try:
                    await asyncio.wait_for(_abort_stop.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

        watcher = asyncio.create_task(_abort_watcher()) if should_abort else None

        cap = getattr(self, "_max_calls_per_file", 0)  # instances built via __new__ in tests lack it
        if cap and total > cap:
            logger.warning(
                f"Context generation for '{file_name}': {total} chunks exceed "
                f"max_calls_per_file={cap}; chunks after the first {cap} get the fallback prefix")

        async def _capped(chunk_text: str) -> str:
            _on_one_done()
            return _safe_fallback_prefix(chunk_text, file_name)

        try:
            tasks = []
            for i, chunk_text in enumerate(chunks):
                if cap and i >= cap:
                    tasks.append(_capped(chunk_text))
                    continue
                tasks.append(self._generate_with_progress(
                    chunk_text, full_document, file_name,
                    client, semaphore, _on_one_done, should_abort,
                ))

            results = await asyncio.gather(*tasks)
        finally:
            _abort_stop.set()
            if watcher is not None:
                try:
                    await watcher
                except Exception:
                    pass
            try:
                client.close()
            except Exception as e:
                logger.debug(f"LLM client close failed (ignored): {e}")

        logger.info(f"Context generation completed: {total} chunks for {file_name}")
        return results

    async def _generate_with_progress(
        self,
        chunk_text: str,
        full_document: str,
        file_name: str,
        client,
        semaphore: asyncio.Semaphore,
        on_done: Callable[[], None],
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Generate one chunk's context and report completion."""
        result = await self._generate_one(
            chunk_text, full_document, file_name, client, semaphore, should_abort
        )
        on_done()
        return result
