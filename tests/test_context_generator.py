"""Unit tests for src/domain/rag/context_generator.py Contextual Retrieval logic.

The LLM call path is covered by pytest-mock stubs (no real requests).
"""

import asyncio
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from src.domain.rag.context_generator import (
    ContextGenerator,
    _detect_scripts,
    _is_script_mismatch,
)


def _make_generator(**overrides) -> ContextGenerator:
    """Build a ContextGenerator that needs no real LLM config (for unit tests)"""
    kwargs = dict(llm_config=None, model="test-model")
    kwargs.update(overrides)
    return ContextGenerator(**kwargs)


# _detect_scripts

def test_detect_scripts_latin():
    """Pure English (including Latin Extended accented letters) returns {'Latin'}"""
    assert _detect_scripts("Hello world") == {"Latin"}
    assert _detect_scripts("café résumé") == {"Latin"}


def test_detect_scripts_cjk():
    """Pure Chinese returns {'CJK'}"""
    assert _detect_scripts("這是一段中文") == {"CJK"}


def test_detect_scripts_japanese_kana_and_kanji():
    """Japanese hiragana/katakana/kanji are detected as Hiragana / Katakana / CJK respectively"""
    assert _detect_scripts("こんにちは") == {"Hiragana"}
    assert _detect_scripts("カタカナ") == {"Katakana"}
    assert _detect_scripts("日本語のテスト") == {"CJK", "Hiragana", "Katakana"}


@pytest.mark.parametrize(
    "text, expected_script",
    [
        ("안녕하세요", "Hangul"),
        ("Привет", "Cyrillic"),
        ("مرحبا", "Arabic"),
        ("שלום", "Hebrew"),
        ("สวัสดี", "Thai"),
        ("नमस्ते", "Devanagari"),
    ],
    ids=["hangul", "cyrillic", "arabic", "hebrew", "thai", "devanagari"],
)
def test_detect_scripts_other_major_scripts(text, expected_script):
    """Korean/Russian/Arabic/Hebrew/Thai/Devanagari are each classified correctly"""
    assert expected_script in _detect_scripts(text)


def test_detect_scripts_ignores_non_alpha():
    """Digits, punctuation, and whitespace are not counted; an empty string returns an empty set"""
    assert _detect_scripts("12345 !?,.()") == set()
    assert _detect_scripts("") == set()


def test_detect_scripts_mixed_text():
    """Mixed Chinese-English returns two scripts"""
    assert _detect_scripts("公司 Q3 revenue 成長") == {"CJK", "Latin"}


# _is_script_mismatch

def test_mismatch_cjk_chunk_latin_prefix():
    """Chinese chunk with a pure-English prefix -> mismatch=True"""
    assert _is_script_mismatch("這是中文段落", "This chunk describes something") is True


def test_no_mismatch_cjk_chunk_cjk_prefix():
    """Chinese chunk with a Chinese prefix (may include Latin/digits) -> False"""
    assert _is_script_mismatch("這是中文段落", "本段說明 Q3 財報重點") is False


def test_no_mismatch_pure_latin_chunk():
    """A pure-English chunk is always False regardless of prefix language (Latin tolerance)"""
    assert _is_script_mismatch("English chunk text", "English prefix") is False
    assert _is_script_mismatch("English chunk text", "中文前綴") is False


def test_mismatch_cjk_chunk_empty_or_numeric_prefix():
    """Chinese chunk with an empty or numeric-only prefix (no non-Latin script) -> True"""
    assert _is_script_mismatch("中文內容", "") is True
    assert _is_script_mismatch("中文內容", "12345") is True


def test_mismatch_mixed_chunk_uses_non_latin_main_script():
    """A mixed Chinese-English chunk is judged by its dominant non-Latin script: pure-English prefix -> True, Chinese prefix -> False"""
    assert _is_script_mismatch("中文 mixed with English", "english only prefix") is True
    assert _is_script_mismatch("中文 mixed with English", "中文前綴") is False


def test_mismatch_hangul_chunk_latin_prefix():
    """A non-Chinese non-Latin script (Korean) is protected the same way -> True"""
    assert _is_script_mismatch("안녕하세요 문서입니다", "English prefix") is True


# ContextGenerator construction / from_config

def test_init_stores_params_and_defaults():
    """__init__ stores the required parameters and applies defaults"""
    gen = _make_generator()
    assert gen._model == "test-model"
    assert gen._max_context_length == 150
    assert gen._max_concurrent == 50
    assert gen._max_doc_chars == 60000
    assert gen._max_tokens == 1024
    assert gen._reasoning_effort is None


def test_from_config_disabled_returns_none():
    """Returns None when unset or enabled=False"""
    assert ContextGenerator.from_config(SimpleNamespace()) is None
    cfg = SimpleNamespace(contextual_retrieval=SimpleNamespace(enabled=False))
    assert ContextGenerator.from_config(cfg) is None


def test_from_config_missing_llm_returns_none():
    """enabled=True but missing llm config returns None"""
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(enabled=True, max_context_length=100),
        llm=None,
    )
    assert ContextGenerator.from_config(cfg) is None


def test_from_config_success_prefers_azure_deployment():
    """A normal config returns an instance; model prefers azure_deployment"""
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(
            enabled=True,
            max_context_length=120,
            max_concurrent=8,
            max_doc_chars=1000,
            max_tokens=256,
            reasoning_effort="low",
        ),
        llm=SimpleNamespace(azure_deployment="gpt-4o-dep", model="gpt-4o"),
    )
    gen = ContextGenerator.from_config(cfg)
    assert isinstance(gen, ContextGenerator)
    assert gen._model == "gpt-4o-dep"
    assert gen._max_context_length == 120
    assert gen._max_concurrent == 8
    assert gen._max_doc_chars == 1000
    assert gen._max_tokens == 256
    assert gen._reasoning_effort == "low"


def test_from_config_model_fallback_and_optional_defaults():
    """When azure_deployment is None, fall back to model; missing optional fields use defaults"""
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(enabled=True, max_context_length=100),
        llm=SimpleNamespace(azure_deployment=None, model="qwen3"),
    )
    gen = ContextGenerator.from_config(cfg)
    assert gen._model == "qwen3"
    assert gen._max_concurrent == 50
    assert gen._max_doc_chars == 60000
    assert gen._max_tokens == 1024
    assert gen._reasoning_effort is None


def test_from_config_missing_required_field_returns_none():
    """A missing required field (max_context_length) has its internal exception swallowed and returns None"""
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(enabled=True),
        llm=SimpleNamespace(azure_deployment=None, model="m"),
    )
    assert ContextGenerator.from_config(cfg) is None


# _create_sync_client (only tests the error and parameter paths that need no network)

def test_create_sync_client_unsupported_provider():
    """An unsupported provider raises ValueError"""
    llm = SimpleNamespace(provider="groq")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        ContextGenerator._create_sync_client(llm)


def test_create_sync_client_vllm_requires_base_url():
    """vllm without base_url raises ValueError"""
    llm = SimpleNamespace(provider="vllm", base_url=None, api_key=None)
    with pytest.raises(ValueError, match="requires base_url"):
        ContextGenerator._create_sync_client(llm)


def test_create_sync_client_vllm_builds_openai_client():
    """vllm with base_url builds an OpenAI client pointing at that URL (a placeholder is used when there is no api_key)"""
    llm = SimpleNamespace(provider="vllm", base_url="http://localhost:8000/v1", api_key=None)
    client = ContextGenerator._create_sync_client(llm)
    assert str(client.base_url).startswith("http://localhost:8000/v1")
    assert client.api_key == "not-needed"


# _call_llm_sync (stub client, no real requests)

class _StubClient:
    """Minimal OpenAI client stub -- records create() kwargs and returns a pre-supplied message"""

    def __init__(self, message):
        self.captured_kwargs = None
        self._message = message
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured_kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


def test_call_llm_sync_returns_content_and_request_params():
    """Returns message.content, passing model / max_tokens / temperature=0"""
    stub = _StubClient(SimpleNamespace(content="生成的上下文"))
    gen = _make_generator(max_tokens=256)

    result = gen._call_llm_sync("some prompt", stub)

    assert result == "生成的上下文"
    assert stub.captured_kwargs["model"] == "test-model"
    assert stub.captured_kwargs["max_tokens"] == 256
    assert stub.captured_kwargs["temperature"] == 0
    assert "reasoning_effort" not in stub.captured_kwargs  # must not be passed when unset


def test_call_llm_sync_never_uses_reasoning_content_and_passes_effort():
    """Empty content stays empty: the chain of thought in reasoning_content must never
    become a prefix; reasoning_effort is passed when set"""
    stub = _StubClient(SimpleNamespace(content="", reasoning_content="We need to think..."))
    gen = _make_generator(reasoning_effort="low")

    result = gen._call_llm_sync("prompt", stub)

    assert not result
    assert stub.captured_kwargs["reasoning_effort"] == "low"


async def test_generate_one_reasoning_only_answer_is_a_failure():
    """A reasoning model that spent its budget thinking yields None, not its thoughts"""
    gen = _make_generator()
    stub = _StubClient(SimpleNamespace(content=None, reasoning_content="Let me analyze the chunk"))

    result = await gen._generate_one(
        "english chunk", "doc", "f.txt", client=stub, semaphore=asyncio.Semaphore(1),
    )

    assert result is None


# _generate_one / generate_prefixes (LLM replaced by a mock)

async def test_generate_one_success(mocker: MockerFixture):
    """A normal LLM response -> the prefix alone; the chunk text is stored separately"""
    gen = _make_generator()
    mocker.patch.object(gen, "_call_llm_sync", return_value="本段說明差旅費核銷規定")

    result = await gen._generate_one(
        "前項所述之情形,不適用本條規定。", "全文...", "差旅辦法.docx",
        client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result == "本段說明差旅費核銷規定"


async def test_generate_one_truncates_long_context(mocker: MockerFixture):
    """When context exceeds max_context_length, it is truncated before concatenation"""
    gen = _make_generator(max_context_length=5)
    mocker.patch.object(gen, "_call_llm_sync", return_value="A" * 20)

    result = await gen._generate_one(
        "english chunk", "doc", "f.txt", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result == "AAAAA"


async def test_generate_one_empty_llm_response_falls_back(mocker: MockerFixture):
    """LLM returns an empty string -> None (no prefix; retried on the next re-index)"""
    gen = _make_generator()
    mocker.patch.object(gen, "_call_llm_sync", return_value="")

    result = await gen._generate_one(
        "chunk 內容", "doc", "報告.docx", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result is None


async def test_generate_one_script_mismatch_is_a_failure(mocker: MockerFixture):
    """A Chinese chunk that gets a pure-English prefix twice -> None (never cached as "")"""
    gen = _make_generator()
    call = mocker.patch.object(gen, "_call_llm_sync", return_value="This chunk is about travel policy")

    result = await gen._generate_one(
        "中文段落內容", "doc", "辦法.docx", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result is None
    assert call.call_count == 2  # retried once


async def test_generate_one_retries_once_after_unusable_answer(mocker: MockerFixture):
    """Empty first answer, good second answer -> the second one is used"""
    gen = _make_generator()
    call = mocker.patch.object(gen, "_call_llm_sync", side_effect=["", "本段說明住宿費上限"])

    result = await gen._generate_one(
        "住宿費每日以二千元為上限。", "doc", "辦法.docx", client=None,
        semaphore=asyncio.Semaphore(1),
    )

    assert result == "本段說明住宿費上限"
    assert call.call_count == 2


async def test_generate_one_retry_is_charged_to_the_daily_budget(mocker: MockerFixture):
    from src.domain.rag.usage_budget import DailyBudget
    budget = DailyBudget()
    gen = _make_generator(max_tokens=10, max_tokens_per_day=10_000, budget=budget)
    mocker.patch.object(gen, "_call_llm_sync", side_effect=["", "ctx"])
    await gen._generate_one("chunk", "doc", "f.txt", client=None, semaphore=asyncio.Semaphore(1))
    one = budget.used("prefix_tokens")
    budget2 = DailyBudget()
    gen2 = _make_generator(max_tokens=10, max_tokens_per_day=10_000, budget=budget2)
    mocker.patch.object(gen2, "_call_llm_sync", return_value="ctx")
    await gen2._generate_one("chunk", "doc", "f.txt", client=None, semaphore=asyncio.Semaphore(1))
    assert one == 2 * budget2.used("prefix_tokens")


async def test_generate_one_llm_exception_falls_back(mocker: MockerFixture):
    """LLM raises an exception -> not leaked, returns None"""
    gen = _make_generator()
    mocker.patch.object(gen, "_call_llm_sync", side_effect=RuntimeError("connection refused"))

    result = await gen._generate_one(
        "chunk text", "doc", "f.txt", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result is None


async def test_generate_one_truncates_long_document_in_prompt(mocker: MockerFixture):
    """Document context longer than max_doc_chars is cut to a window, marked with an ellipsis"""
    gen = _make_generator(max_doc_chars=10)
    mock_call = mocker.patch.object(gen, "_call_llm_sync", return_value="ctx")

    await gen._generate_one(
        "chunk", "D" * 100, "f.txt", client=None, semaphore=asyncio.Semaphore(1),
    )

    prompt = mock_call.call_args[0][0]
    assert "D" * 10 + "…" in prompt
    assert "D" * 11 not in prompt


async def test_generate_batch_order_and_length(mocker: MockerFixture):
    """generate_prefixes returns a list the same length as chunks, matching order one-to-one"""
    gen = _make_generator()
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    mocker.patch.object(gen, "_call_llm_sync", return_value="CTX")

    chunks = ["chunk one", "chunk two", "chunk three"]
    results = await gen.generate_prefixes(chunks, "full document", "file.txt")

    assert results == ["CTX"] * len(chunks)


# Cost guards (rag.contextual_retrieval.max_calls_per_file / max_tokens_per_day)

async def test_generate_prefixes_caps_llm_calls_per_file(mocker: MockerFixture):
    """Chunks past max_calls_per_file get no prefix (None) without an LLM call"""
    from src.domain.rag.usage_budget import DailyBudget
    gen = _make_generator(max_calls_per_file=2, budget=DailyBudget())
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    call = mocker.patch.object(gen, "_call_llm_sync", return_value="CTX")
    progress = []
    results = await gen.generate_prefixes(["a", "b", "c"], "doc", "f.txt",
                                          on_progress=lambda d, t: progress.append((d, t)))
    assert results == ["CTX", "CTX", None]  # None: the next re-index retries it
    assert call.call_count == 2 and progress[-1] == (3, 3)


async def test_daily_token_budget_falls_back_when_exhausted(mocker: MockerFixture):
    """Once today's estimated prefix tokens are spent, chunks get no prefix (None)"""
    from src.domain.rag.usage_budget import DailyBudget
    budget = DailyBudget()
    gen = _make_generator(max_tokens=100, max_tokens_per_day=1, budget=budget, max_concurrent=1)
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    call = mocker.patch.object(gen, "_call_llm_sync", return_value="CTX")
    results = await gen.generate_prefixes(["a", "b"], "doc", "f.txt")
    assert results == [None, None] and call.call_count == 0
    assert budget.used("prefix_tokens") == 0


def test_from_config_passes_cost_guards():
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(enabled=True, max_context_length=100,
                                             max_calls_per_file=7, max_tokens_per_day=9000),
        llm=SimpleNamespace(azure_deployment=None, model="m"),
    )
    gen = ContextGenerator.from_config(cfg)
    assert gen._max_calls_per_file == 7 and gen._max_tokens_per_day == 9000


# Process-wide LLM concurrency (rag.contextual_retrieval.max_concurrent_global)

async def test_global_cap_bounds_concurrent_files(mocker: MockerFixture):
    """Two files indexed at once, 5 per file allowed, but only 2 requests in flight overall"""
    import threading
    import time

    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def _slow_call(prompt, client):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.03)
        with lock:
            state["now"] -= 1
        return "CTX"

    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    gens = []
    for _ in range(2):
        g = _make_generator(max_concurrent=5, max_concurrent_global=2)
        mocker.patch.object(g, "_call_llm_sync", side_effect=_slow_call)
        gens.append(g)
    results = await asyncio.gather(*(g.generate_prefixes(["a", "b", "c", "d"], "doc", "f.txt")
                                     for g in gens))
    assert results == [["CTX"] * 4, ["CTX"] * 4]
    assert state["peak"] == 2


def test_global_semaphore_is_shared_and_replaced_on_change():
    from src.domain.rag import context_generator as cg

    a = cg._global_semaphore(3)
    assert cg._global_semaphore(3) is a
    b = cg._global_semaphore(4)
    assert b is not a and cg._global_semaphore(4) is b


def test_from_config_reads_global_cap():
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(enabled=True, max_context_length=100,
                                             max_concurrent_global=7),
        llm=SimpleNamespace(azure_deployment=None, model="m"),
    )
    assert ContextGenerator.from_config(cfg)._max_concurrent_global == 7
    cfg.contextual_retrieval = SimpleNamespace(enabled=True, max_context_length=100)
    assert ContextGenerator.from_config(cfg)._max_concurrent_global == 16


# Prefix trimming and the CJK-aware budget estimate (L3 / L4)

def test_trim_prefix_ends_at_last_sentence_within_limit():
    from src.domain.rag.context_generator import trim_prefix
    text = "本段說明差旅費之核銷流程。另說明住宿費每日上限與超額部分之處理方式，以及膳雜費計算。"
    out = trim_prefix(text, 30)
    assert out == "本段說明差旅費之核銷流程。"
    en = "This section covers travel claims. It also lists the nightly hotel cap and meals."
    assert trim_prefix(en, 60) == "This section covers travel claims."


def test_trim_prefix_falls_back_to_clause_or_word_break():
    from src.domain.rag.context_generator import trim_prefix
    text = "本段說明差旅費之核銷流程，另說明住宿費每日上限與超額部分之處理方式"
    assert trim_prefix(text, 20) == "本段說明差旅費之核銷流程"
    en = "Travel claim rules for domestic and overseas trips including hotel caps"
    out = trim_prefix(en, 40)
    assert len(out) <= 40 and en.startswith(out) and not out.endswith(" ")
    assert en[len(out)] == " "  # cut at a word break


def test_trim_prefix_short_text_and_hard_cut():
    from src.domain.rag.context_generator import trim_prefix
    assert trim_prefix("short.", 150) == "short."
    assert trim_prefix("A" * 20, 5) == "AAAAA"
    # a sentence end in the first 40% is not used
    assert trim_prefix("Hi. " + "x" * 50, 20) == "Hi. " + "x" * 16


async def test_generate_one_trims_at_sentence_end(mocker: MockerFixture):
    gen = _make_generator(max_context_length=30)
    mocker.patch.object(gen, "_call_llm_sync",
                        return_value="本段說明差旅費之核銷流程。另說明住宿費每日上限與超額部分之處理方式。")
    result = await gen._generate_one("住宿費上限。", "doc", "f.docx", client=None,
                                     semaphore=asyncio.Semaphore(1))
    assert result == "本段說明差旅費之核銷流程。"


def test_estimate_llm_tokens_counts_cjk_per_char():
    from src.domain.rag.context_generator import estimate_llm_tokens
    assert estimate_llm_tokens("一二三四五六七八") == 8   # len // 4 would say 2
    assert estimate_llm_tokens("abcdefgh") == 2
    assert estimate_llm_tokens("中文abcd") == 3
    assert estimate_llm_tokens("") == 0


async def test_daily_budget_charges_cjk_prompts_fully(mocker: MockerFixture):
    from src.domain.rag.context_generator import estimate_llm_tokens
    from src.domain.rag.usage_budget import DailyBudget
    budget = DailyBudget()
    gen = _make_generator(max_tokens=0, max_tokens_per_day=10**9, budget=budget)
    call = mocker.patch.object(gen, "_call_llm_sync", return_value="本段說明")
    chunk = "住宿費每日以新臺幣二千元為上限。" * 20
    await gen._generate_one(chunk, "doc", "f.docx", client=None, semaphore=asyncio.Semaphore(1))
    prompt = call.call_args[0][0]
    assert budget.used("prefix_tokens") == estimate_llm_tokens(prompt)
    assert budget.used("prefix_tokens") > len(prompt) // 4 + len(chunk) // 2
