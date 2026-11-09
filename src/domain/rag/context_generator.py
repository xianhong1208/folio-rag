
"""Contextual Retrieval — Context Generator

Generates a context prefix for each chunk, resolving the semantic ambiguity a chunk has once
detached from its source document.
Reference: Anthropic "Introducing Contextual Retrieval" (2024-09)

Usage:
    generator = ContextGenerator.from_config(rag_config)
    prefixes = await generator.generate_prefixes(
        chunks=["The provisions above do not apply in this case.", ...],
        full_document="Full text of the travel policy...",
        file_name="travel_policy.docx",
    )

Each prefix is stored next to the chunk (metadata.context_prefix), never inside the
chunk's displayed text; see chunk_identity for how it joins the retrieval text.

Performance:
    Uses a sync OpenAI client + asyncio.to_thread to call the LLM concurrently, throttled by an
    asyncio.Semaphore, maximizing vLLM continuous-batching efficiency. The sync client avoids the
    AsyncOpenAI event-loop issues seen under PyInstaller-packaged environments.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
from typing import AsyncIterator, Callable, Dict, Optional, List

from src.log import get_api_logger
from src.domain.rag.context_scope import SECTION_WINDOW_CHARS, locate_window, window
from src.domain.rag.han_script import document_language, fix_simplified, strip_invisible

logger = get_api_logger()

# The prefix must be in the chunk's own language and script, so that it lands in the
# same region of the embedding space as the chunk (an English prefix on a Chinese chunk
# drifts and loses same-language queries). The instructions are written in English and
# carry no examples in any particular language: a detected document language is named
# explicitly instead ({language_rule}), which also stops models from slipping into
# English on CJK text. Document context comes first and the chunk last, so chunks of
# one section share a prompt prefix (vLLM prefix cache).
_CONTEXT_PROMPT = """\
You write one short context line that situates a chunk of a document, so that a \
search engine can understand the chunk on its own.

Rules:
- Write in the same language and script as the chunk.{language_rule}
- One or two sentences, at most {max_length} characters.
- Say which part of the document the chunk belongs to and what it is about. Name the \
document, product, organisation, person or topic when the context makes it clear.
- Identify the speaker when the chunk is a transcript and the speaker is obvious.
- Do not restate the chunk. No quotation marks, no preamble such as "This chunk".

Document: {file_name}

<outline>
{outline}
</outline>

<section>
{section}
</section>

<chunk>
{chunk_text}
</chunk>

Context line:"""


def _language_rule(language: Optional[str]) -> str:
    if not language:
        return ""
    rule = (f" The document is written in {language}: when the chunk has little prose "
            f"of its own (numbers, code, a table), write in {language}.")
    if language == "Traditional Chinese":
        rule += " Use Traditional characters only, never Simplified ones."
    return rule


# A logographic or syllabic character carries about a word; an alphabetic letter
# about a quarter of one. Weighing letters this way tells which script a mixed chunk
# is mostly written in.
_WORD_SCRIPTS = {"CJK", "Hiragana", "Katakana", "Hangul"}


def _script_weights(text: str) -> Dict[str, float]:
    """How much of ``text`` each major script carries, in rough words."""
    weights: Dict[str, float] = {}
    for script in _scripts_of(text):
        weights[script] = weights.get(script, 0.0) + (1.0 if script in _WORD_SCRIPTS else 0.25)
    return weights


def _detect_scripts(text: str) -> set:
    """Detect the set of major Unicode scripts present in text, deciding directly
    from the Unicode block (no hardcoded language). Returns e.g. {"Latin", "CJK"}.
    """
    return set(_scripts_of(text))


def _scripts_of(text: str):
    """The script of each letter in ``text`` (one item per letter)."""
    scripts = []
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x0080:
            scripts.append("Latin")
        elif 0x00C0 <= cp < 0x0250:
            scripts.append("Latin")  # Latin Extended
        elif 0x0400 <= cp < 0x0500:
            scripts.append("Cyrillic")
        elif 0x0590 <= cp < 0x0600:
            scripts.append("Hebrew")
        elif 0x0600 <= cp < 0x0700:
            scripts.append("Arabic")
        elif 0x0900 <= cp < 0x0980:
            scripts.append("Devanagari")
        elif 0x0E00 <= cp < 0x0E80:
            scripts.append("Thai")
        elif 0x3040 <= cp < 0x30A0:
            scripts.append("Hiragana")
        elif 0x30A0 <= cp < 0x3100:
            scripts.append("Katakana")
        elif 0xAC00 <= cp < 0xD7B0:
            scripts.append("Hangul")
        elif 0x4E00 <= cp < 0xA000:
            scripts.append("CJK")
    return scripts


def _is_script_mismatch(chunk_text: str, candidate_prefix: str) -> bool:
    """Return True if the prefix's main script mismatches the chunk's (avoids vector drift).

    A chunk mostly written in a non-Latin script (CJK / Hangul / Arabic / Cyrillic …)
    with an all-Latin prefix is a mismatch. "Mostly" matters: a mainly English chunk
    with one Chinese line used to count as Chinese, and its correct English prefix was
    thrown away. Incidental Latin alphanumerics never make a mismatch.
    """
    weights = _script_weights(chunk_text)
    if not weights:
        return False
    chunk_main = max(weights, key=weights.get)
    prefix_main = _detect_scripts(candidate_prefix) - {"Latin"}
    return chunk_main != "Latin" and not prefix_main


def estimate_llm_tokens(text: str) -> int:
    """Cost estimate for the daily budget: one token per CJK / kana / hangul character,
    one per four other characters. ``len(text) // 4`` undercounted Chinese ~3x."""
    wide = sum(
        1 for ch in text or ""
        if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
        or "\uac00" <= ch <= "\ud7af" or "\uf900" <= ch <= "\ufaff"
    )
    return wide + (len(text or "") - wide + 3) // 4


_PREFIX_SENTENCE_END = re.compile(r"[。！？；]|[.!?](?=\s|$)")
_PREFIX_SOFT_BREAK = re.compile(r"[，、：,;:]|\s")


def trim_prefix(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters at the last sentence end, else at
    the last clause / word break, else hard. A break in the first 40% is ignored (it
    would throw most of the prefix away)."""
    if len(text) <= limit:
        return text
    floor = int(limit * 0.4)
    head = text[:limit + 1]  # a terminator right at the limit still counts
    ends = [m.end() for m in _PREFIX_SENTENCE_END.finditer(head) if floor <= m.end() <= limit]
    if ends:
        return text[:ends[-1]].strip()
    breaks = [m.start() for m in _PREFIX_SOFT_BREAK.finditer(head) if floor <= m.start() <= limit]
    if breaks:
        return text[:breaks[-1]].rstrip(" ，、：,;:")
    return text[:limit]


# Process-wide cap on in-flight prefix requests. The per-call semaphore only bounds
# one file, so N files indexed at once sent N x max_concurrent requests and overloaded
# the LLM server. Jobs may run on different event loops (worker threads), so the cap
# is a threading semaphore, polled from the loop with a short sleep rather than
# blocking: a parked executor thread per waiting request would starve every other
# to_thread user (embedding, DB writes).
DEFAULT_MAX_CONCURRENT_GLOBAL = 16
_global_lock = threading.Lock()
_global_slots: Dict[int, threading.BoundedSemaphore] = {}


def _global_semaphore(n: int) -> threading.BoundedSemaphore:
    n = max(1, int(n))
    with _global_lock:
        if n not in _global_slots:
            _global_slots.clear()  # a changed setting replaces the limiter for new requests
            _global_slots[n] = threading.BoundedSemaphore(n)
        return _global_slots[n]


@contextlib.asynccontextmanager
async def _global_slot(n: int) -> AsyncIterator[None]:
    sem = _global_semaphore(n)
    delay = 0.005
    while not sem.acquire(blocking=False):
        await asyncio.sleep(delay)
        delay = min(delay * 2, 0.1)
    try:
        yield
    finally:
        sem.release()


# Part of every prefix's cache key: bump it whenever the prompt changes so that an
# incremental re-index regenerates prefixes written by an older prompt.
PROMPT_VERSION = "3"  # 3: drop prefixes cached from the old reasoning_content fallback; 2: outline + own section, language-neutral prompt, script repair


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
        max_concurrent_global: int = DEFAULT_MAX_CONCURRENT_GLOBAL,
    ):
        self._llm_config = llm_config
        self._model = model
        self._max_context_length = max_context_length
        self._max_concurrent = max_concurrent
        self._max_doc_chars = max_doc_chars
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._max_concurrent_global = max_concurrent_global
        # Cost guards (0 = unlimited): chunks past a cap get the fallback prefix, no LLM call.
        self._max_calls_per_file = max_calls_per_file
        self._max_tokens_per_day = max_tokens_per_day
        if budget is None:
            from src.domain.rag.usage_budget import BUDGET as budget
        self._budget = budget
        logger.info(
            f"ContextGenerator initialized (model={model}, max_length={max_context_length}, "
            f"max_concurrent={max_concurrent} (process-wide {max_concurrent_global}), max_doc_chars={max_doc_chars}, "
            f"max_tokens={max_tokens}, reasoning_effort={reasoning_effort})"
        )

    @property
    def max_doc_chars(self) -> int:
        """Upper bound on the document text given to the model per chunk."""
        return self._max_doc_chars

    @property
    def cache_key(self) -> str:
        """Identity of what produces a prefix (model + prompt + length limit). A stored
        prefix is reused on re-index only when its key matches."""
        return f"{self._model}|prompt={PROMPT_VERSION}|max={self._max_context_length}"

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
                max_concurrent_global=int(getattr(cr_config, 'max_concurrent_global', 0)
                                          or DEFAULT_MAX_CONCURRENT_GLOBAL),
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
        # Only the answer. A reasoning model that ran out of completion budget leaves
        # ``content`` empty and its chain of thought in ``reasoning_content``; using
        # that as the prefix embedded and indexed the model's scratch work.
        return msg.content

    def _admit_tokens(self, prompt: str) -> bool:
        """Charge an estimated cost (CJK-aware prompt estimate + max_tokens) to today's prefix budget."""
        limit = getattr(self, "_max_tokens_per_day", 0)
        if not limit:
            return True
        estimate = estimate_llm_tokens(prompt) + self._max_tokens
        if self._budget.try_consume("prefix_tokens", estimate, limit):
            return True
        if not getattr(self, "_budget_warned", False):
            self._budget_warned = True
            logger.warning(
                f"Contextual-prefix daily token budget ({limit}) exhausted — "
                "remaining chunks get no prefix until UTC midnight (a later re-index fills them in)")
        return False

    def _postprocess(self, raw_content: Optional[str], chunk_text: str,
                     script_reference: str) -> Optional[str]:
        """Clean the model's answer into a prefix; None when it is unusable (empty,
        or in another script than the chunk)."""
        if not raw_content or not raw_content.strip():
            return None
        # Zero-width characters split keyword tokens ("ROC\u200bm"); Simplified
        # characters in a Traditional document never match Traditional queries.
        context = " ".join(strip_invisible(raw_content).split())
        context = fix_simplified(context, script_reference)
        # A script-mismatched prefix+chunk drifts in embedding space and tanks the
        # chunk's hit rate on same-language queries.
        if not context or _is_script_mismatch(chunk_text, context):
            return None
        return trim_prefix(context, self._max_context_length) or None

    async def _generate_one(
        self,
        chunk_text: str,
        context_text: str,
        file_name: str,
        client,
        semaphore: asyncio.Semaphore,
        should_abort: Optional[Callable[[], bool]] = None,
        outline: str = "",
        language: Optional[str] = None,
        script_reference: str = "",
    ) -> Optional[str]:
        """Generate the context prefix for a single chunk.

        ``context_text`` is the part of the document the model reads for this chunk
        (its section, see context_scope); ``outline`` the document's heading paths;
        ``script_reference`` the document text used to repair Simplified drift.

        Returns the prefix, or None when no usable answer was obtained (empty or
        wrong-script answer twice, error, budget, abort): a None prefix is never cached,
        so a later re-index tries again. There is no file-name fallback: the chunk's
        breadcrumb already carries the file name.

        When should_abort holds, skip the LLM — after an abort the whole batch's results
        are discarded by the indexer (raises IndexingAbortedError), so the goal here is
        only to let gather finish quickly without wasting LLM calls.
        """
        try:
            if should_abort and should_abort():
                return None

            section = context_text or ""
            if len(section) > self._max_doc_chars:  # safety net; callers pre-window
                section = window(section, 0, 0, self._max_doc_chars)

            prompt = _CONTEXT_PROMPT.format(
                max_length=self._max_context_length,
                language_rule=_language_rule(language),
                file_name=file_name,
                outline=outline or "(no headings)",
                section=section,
                chunk_text=chunk_text,
            )
            # An empty answer or one in the wrong script is a failure, not a prefix:
            # ask once more (a reasoning model's use of its completion budget varies),
            # then give up with None so the next re-index retries it.
            for attempt in (1, 2):
                if not self._admit_tokens(prompt):
                    return None  # daily budget spent: no prefix now, a later re-index retries
                limit = getattr(self, "_max_concurrent_global", DEFAULT_MAX_CONCURRENT_GLOBAL)
                async with semaphore, _global_slot(limit):
                    # A chunk queued behind the semaphores may have waited a while; re-check once it acquires the slot
                    if should_abort and should_abort():
                        return None
                    raw_content = await asyncio.to_thread(self._call_llm_sync, prompt, client)
                context = self._postprocess(raw_content, chunk_text, script_reference or section)
                if context:
                    return context
                logger.warning(
                    f"Unusable context prefix for chunk in '{file_name}' "
                    f"(attempt {attempt}/2): "
                    + ("empty answer" if not (raw_content or "").strip()
                       else f"script mismatch, preview={(raw_content or '')[:60]!r}")
                )
            return None

        except Exception as e:
            logger.warning(f"Context generation failed for chunk in '{file_name}': {e}")
            return None

    async def generate_prefixes(
        self,
        chunks: List[str],
        full_document: str,
        file_name: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
        contexts: Optional[List[str]] = None,
        outline: str = "",
        language: Optional[str] = None,
    ) -> List[Optional[str]]:
        """Generate contextual prefixes for a batch of chunks (concurrent LLM calls).

        Sync OpenAI client + asyncio.to_thread for concurrency; asyncio.Semaphore for throttling.

        Args:
            chunks: List of chunk texts to prefix.
            full_document: Full document content. Used for the script repair, for the
                language when ``language`` is None, and — only when ``contexts`` is not
                given — as a window centred on each chunk.
            file_name: File-name tag (for logging + fallback).
            on_progress: ``(done, total)`` progress callback; invoked every 10 completed chunks or
                on the last one (throttled, so job status can show progress). None = silent.
            should_abort: When it returns True, remaining chunks skip the LLM to finish quickly
                (cooperative cancellation for a file deleted mid-run). The caller guarantees this
                callable won't raise; abort adjudication and result discarding are the indexer's
                job, and this method returns normally.
            contexts: per chunk, the document text the model reads (its section, see
                context_scope.section_texts).
            outline: the document's heading paths (context_scope.document_outline).
            language: the document's language name, e.g. "Traditional Chinese".

        Returns:
            A list the same length as chunks: each chunk's prefix, or None when no
            usable answer was obtained (see _generate_one).
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

        width = min(self._max_doc_chars, SECTION_WINDOW_CHARS)
        if contexts is None:
            contexts = [locate_window(full_document or "", c, width) for c in chunks]
        if language is None:
            language = document_language(full_document or "\n".join(contexts))
        script_reference = (full_document or "")[:20000]

        # Cost guard: past max_calls_per_file a chunk gets no prefix (None, like a failed
        # call), so the next incremental re-index fills it in instead of keeping a stub.
        cap = getattr(self, "_max_calls_per_file", 0)  # instances built via __new__ in tests lack it
        if cap and total > cap:
            logger.warning(
                f"Context generation for '{file_name}': {total} chunks exceed "
                f"max_calls_per_file={cap}; chunks after the first {cap} get no prefix "
                "this time (a later re-index fills them in)")

        async def _capped() -> None:
            _on_one_done()
            return None

        try:
            tasks = []
            for i, (chunk_text, context_text) in enumerate(zip(chunks, contexts)):
                if cap and i >= cap:
                    tasks.append(_capped())
                    continue
                tasks.append(self._generate_with_progress(
                    chunk_text, context_text, file_name,
                    client, semaphore, _on_one_done, should_abort,
                    outline, language, script_reference,
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
        context_text: str,
        file_name: str,
        client,
        semaphore: asyncio.Semaphore,
        on_done: Callable[[], None],
        should_abort: Optional[Callable[[], bool]] = None,
        outline: str = "",
        language: Optional[str] = None,
        script_reference: str = "",
    ) -> Optional[str]:
        """Generate one chunk's context and report completion."""
        result = await self._generate_one(
            chunk_text, context_text, file_name, client, semaphore, should_abort,
            outline=outline, language=language, script_reference=script_reference,
        )
        on_done()
        return result
