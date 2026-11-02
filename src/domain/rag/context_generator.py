
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

import os
import asyncio
from typing import Callable, Optional, List

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


# Part of every prefix's cache key: bump it whenever the prompt changes so that an
# incremental re-index regenerates prefixes written by an older prompt.
PROMPT_VERSION = "2"  # 2: outline + own section, language-neutral prompt, script repair


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
    ):
        self._llm_config = llm_config
        self._model = model
        self._max_context_length = max_context_length
        self._max_concurrent = max_concurrent
        self._max_doc_chars = max_doc_chars
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        logger.info(
            f"ContextGenerator initialized (model={model}, max_length={max_context_length}, "
            f"max_concurrent={max_concurrent}, max_doc_chars={max_doc_chars}, "
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

        llm_config = getattr(rag_config, 'llm', None)
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
            )
        except Exception as e:
            logger.error(f"Failed to create ContextGenerator: {e}")
            return None

    @staticmethod
    def _create_sync_client(llm_config):
        """Build a sync OpenAI client (supports Azure / OpenAI / vLLM / Ollama)."""
        from openai import OpenAI

        provider = llm_config.provider.lower()

        if provider == "azure":
            from openai import AzureOpenAI

            api_key = llm_config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Azure LLM api_key is required")

            azure_endpoint = llm_config.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError("Azure endpoint (base_url) is required")

            return AzureOpenAI(
                api_key=api_key,
                api_version=llm_config.api_version or "2024-12-01-preview",
                azure_endpoint=azure_endpoint,
            )

        elif provider in ("vllm", "ollama"):
            base_url = llm_config.base_url
            if not base_url:
                raise ValueError(f"{provider} requires base_url (e.g., http://localhost:8000/v1)")
            api_key = llm_config.api_key or "not-needed"

            return OpenAI(api_key=api_key, base_url=base_url)

        elif provider == "openai":
            api_key = llm_config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI api_key is required")
            return OpenAI(api_key=api_key)

        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Supported: azure, openai, vllm, ollama"
            )

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

        Returns the prefix, "" when the model's answer was unusable (wrong script), or
        None when no answer was obtained (empty response, error, abort) so a later
        re-index can try again. There is no file-name fallback any more: the chunk's
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

            async with semaphore:
                # A chunk queued behind the semaphore may have waited a while; re-check once it acquires the slot
                if should_abort and should_abort():
                    return None
                raw_content = await asyncio.to_thread(self._call_llm_sync, prompt, client)

            if not raw_content:
                logger.warning(
                    f"LLM returned empty content for chunk in '{file_name}'; no prefix"
                )
                return None

            # Zero-width characters split keyword tokens ("ROC\u200bm"); Simplified
            # characters in a Traditional document never match Traditional queries.
            context = " ".join(strip_invisible(raw_content).split())
            context = fix_simplified(context, script_reference or section)

            # A script-mismatched prefix+chunk drifts in bge-m3 space and tanks the
            # chunk's hit rate on same-language queries; prefer the fallback instead.
            if _is_script_mismatch(chunk_text, context):
                logger.warning(
                    f"LLM produced script-mismatched prefix for chunk in '{file_name}',"
                    f" discarding. context_preview={context[:60]!r}"
                )
                return ""

            if len(context) > self._max_context_length:
                context = context[:self._max_context_length]

            return context

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
            A list the same length as chunks: each chunk's prefix, "" when the answer was
            discarded, None when no answer was obtained (see _generate_one).
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

        try:
            tasks = []
            for chunk_text, context_text in zip(chunks, contexts):
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
