"""Unit tests for src/domain/rag/context_generator.py Contextual Retrieval logic.

The LLM call path is covered by pytest-mock stubs (no real requests).
Related docs: docs/testing/specs/SPEC-context-generator.md, docs/testing/test-cases/TC-context-generator.md
"""

import asyncio
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from src.domain.rag.context_generator import (
    ContextGenerator,
    _detect_scripts,
    _is_script_mismatch,
    _safe_fallback_prefix,
)


def _make_generator(**overrides) -> ContextGenerator:
    """Build a ContextGenerator that needs no real LLM config (for unit tests)"""
    kwargs = dict(llm_config=None, model="test-model")
    kwargs.update(overrides)
    return ContextGenerator(**kwargs)


# _detect_scripts

def test_detect_scripts_latin():
    """Pure English (including Latin Extended accented letters) returns {'Latin'} (TC-ctxgen-01)"""
    assert _detect_scripts("Hello world") == {"Latin"}
    assert _detect_scripts("café résumé") == {"Latin"}


def test_detect_scripts_cjk():
    """Pure Chinese returns {'CJK'} (TC-ctxgen-02)"""
    assert _detect_scripts("這是一段中文") == {"CJK"}


def test_detect_scripts_japanese_kana_and_kanji():
    """Japanese hiragana/katakana/kanji are detected as Hiragana / Katakana / CJK respectively (TC-ctxgen-03)"""
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
    """Korean/Russian/Arabic/Hebrew/Thai/Devanagari are each classified correctly (TC-ctxgen-04)"""
    assert expected_script in _detect_scripts(text)


def test_detect_scripts_ignores_non_alpha():
    """Digits, punctuation, and whitespace are not counted; an empty string returns an empty set (TC-ctxgen-05)"""
    assert _detect_scripts("12345 !?,.()") == set()
    assert _detect_scripts("") == set()


def test_detect_scripts_mixed_text():
    """Mixed Chinese-English returns two scripts (TC-ctxgen-06)"""
    assert _detect_scripts("公司 Q3 revenue 成長") == {"CJK", "Latin"}


# _is_script_mismatch

def test_mismatch_cjk_chunk_latin_prefix():
    """Chinese chunk with a pure-English prefix -> mismatch=True (TC-ctxgen-07)"""
    assert _is_script_mismatch("這是中文段落", "This chunk describes something") is True


def test_no_mismatch_cjk_chunk_cjk_prefix():
    """Chinese chunk with a Chinese prefix (may include Latin/digits) -> False (TC-ctxgen-08)"""
    assert _is_script_mismatch("這是中文段落", "本段說明 Q3 財報重點") is False


def test_no_mismatch_pure_latin_chunk():
    """A pure-English chunk is always False regardless of prefix language (Latin tolerance) (TC-ctxgen-09)"""
    assert _is_script_mismatch("English chunk text", "English prefix") is False
    assert _is_script_mismatch("English chunk text", "中文前綴") is False


def test_mismatch_cjk_chunk_empty_or_numeric_prefix():
    """Chinese chunk with an empty or numeric-only prefix (no non-Latin script) -> True (TC-ctxgen-10)"""
    assert _is_script_mismatch("中文內容", "") is True
    assert _is_script_mismatch("中文內容", "12345") is True


def test_mismatch_mixed_chunk_uses_non_latin_main_script():
    """A mixed Chinese-English chunk is judged by its dominant non-Latin script: pure-English prefix -> True, Chinese prefix -> False (TC-ctxgen-11)"""
    assert _is_script_mismatch("中文 mixed with English", "english only prefix") is True
    assert _is_script_mismatch("中文 mixed with English", "中文前綴") is False


def test_mismatch_hangul_chunk_latin_prefix():
    """A non-Chinese non-Latin script (Korean) is protected the same way -> True (TC-ctxgen-12)"""
    assert _is_script_mismatch("안녕하세요 문서입니다", "English prefix") is True


# _safe_fallback_prefix

def test_safe_fallback_prefix_format():
    """Returns the '{file_name}\\n\\n{chunk_text}' format (TC-ctxgen-13)"""
    assert _safe_fallback_prefix("chunk 內容", "報告.docx") == "報告.docx\n\nchunk 內容"


# ContextGenerator construction / from_config

def test_init_stores_params_and_defaults():
    """__init__ stores the required parameters and applies defaults (TC-ctxgen-14)"""
    gen = _make_generator()
    assert gen._model == "test-model"
    assert gen._max_context_length == 150
    assert gen._max_concurrent == 50
    assert gen._max_doc_chars == 60000
    assert gen._max_tokens == 1024
    assert gen._reasoning_effort is None


def test_from_config_disabled_returns_none():
    """Returns None when unset or enabled=False (TC-ctxgen-15)"""
    assert ContextGenerator.from_config(SimpleNamespace()) is None
    cfg = SimpleNamespace(contextual_retrieval=SimpleNamespace(enabled=False))
    assert ContextGenerator.from_config(cfg) is None


def test_from_config_missing_llm_returns_none():
    """enabled=True but missing llm config returns None (TC-ctxgen-16)"""
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(enabled=True, max_context_length=100),
        llm=None,
    )
    assert ContextGenerator.from_config(cfg) is None


def test_from_config_success_prefers_azure_deployment():
    """A normal config returns an instance; model prefers azure_deployment (TC-ctxgen-17)"""
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
    """When azure_deployment is None, fall back to model; missing optional fields use defaults (TC-ctxgen-18)"""
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
    """A missing required field (max_context_length) has its internal exception swallowed and returns None (TC-ctxgen-19)"""
    cfg = SimpleNamespace(
        contextual_retrieval=SimpleNamespace(enabled=True),
        llm=SimpleNamespace(azure_deployment=None, model="m"),
    )
    assert ContextGenerator.from_config(cfg) is None


# _create_sync_client (only tests the error and parameter paths that need no network)

def test_create_sync_client_unsupported_provider():
    """An unsupported provider raises ValueError (TC-ctxgen-20)"""
    llm = SimpleNamespace(provider="groq")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        ContextGenerator._create_sync_client(llm)


def test_create_sync_client_vllm_requires_base_url():
    """vllm without base_url raises ValueError (TC-ctxgen-21)"""
    llm = SimpleNamespace(provider="vllm", base_url=None, api_key=None)
    with pytest.raises(ValueError, match="requires base_url"):
        ContextGenerator._create_sync_client(llm)


def test_create_sync_client_vllm_builds_openai_client():
    """vllm with base_url builds an OpenAI client pointing at that URL (a placeholder is used when there is no api_key) (TC-ctxgen-22)"""
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
    """Returns message.content, passing model / max_tokens / temperature=0 (TC-ctxgen-23)"""
    stub = _StubClient(SimpleNamespace(content="生成的上下文"))
    gen = _make_generator(max_tokens=256)

    result = gen._call_llm_sync("some prompt", stub)

    assert result == "生成的上下文"
    assert stub.captured_kwargs["model"] == "test-model"
    assert stub.captured_kwargs["max_tokens"] == 256
    assert stub.captured_kwargs["temperature"] == 0
    assert "reasoning_effort" not in stub.captured_kwargs  # must not be passed when unset


def test_call_llm_sync_reasoning_content_fallback_and_effort_param():
    """When content is empty, fall back to reasoning_content; reasoning_effort is passed when set (TC-ctxgen-24)"""
    stub = _StubClient(SimpleNamespace(content="", reasoning_content="推理輸出"))
    gen = _make_generator(reasoning_effort="low")

    result = gen._call_llm_sync("prompt", stub)

    assert result == "推理輸出"
    assert stub.captured_kwargs["reasoning_effort"] == "low"


# _generate_one / generate_batch (LLM replaced by a mock)

async def test_generate_one_success(mocker: MockerFixture):
    """A normal LLM response -> '{context}\\n\\n{chunk}' (TC-ctxgen-25)"""
    gen = _make_generator()
    mocker.patch.object(gen, "_call_llm_sync", return_value="本段說明差旅費核銷規定")

    result = await gen._generate_one(
        "前項所述之情形,不適用本條規定。", "全文...", "差旅辦法.docx",
        client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result == "本段說明差旅費核銷規定\n\n前項所述之情形,不適用本條規定。"


async def test_generate_one_truncates_long_context(mocker: MockerFixture):
    """When context exceeds max_context_length, it is truncated before concatenation (TC-ctxgen-26)"""
    gen = _make_generator(max_context_length=5)
    mocker.patch.object(gen, "_call_llm_sync", return_value="A" * 20)

    result = await gen._generate_one(
        "english chunk", "doc", "f.txt", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result == "AAAAA\n\nenglish chunk"


async def test_generate_one_empty_llm_response_falls_back(mocker: MockerFixture):
    """LLM returns an empty string -> safe fallback prefix (TC-ctxgen-27)"""
    gen = _make_generator()
    mocker.patch.object(gen, "_call_llm_sync", return_value="")

    result = await gen._generate_one(
        "chunk 內容", "doc", "報告.docx", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result == "報告.docx\n\nchunk 內容"


async def test_generate_one_script_mismatch_falls_back(mocker: MockerFixture):
    """A Chinese chunk that gets a pure-English prefix -> discard it and use the fallback (TC-ctxgen-28)"""
    gen = _make_generator()
    mocker.patch.object(gen, "_call_llm_sync", return_value="This chunk is about travel policy")

    result = await gen._generate_one(
        "中文段落內容", "doc", "辦法.docx", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result == "辦法.docx\n\n中文段落內容"


async def test_generate_one_llm_exception_falls_back(mocker: MockerFixture):
    """LLM raises an exception -> not leaked, returns the fallback (TC-ctxgen-29)"""
    gen = _make_generator()
    mocker.patch.object(gen, "_call_llm_sync", side_effect=RuntimeError("connection refused"))

    result = await gen._generate_one(
        "chunk text", "doc", "f.txt", client=None, semaphore=asyncio.Semaphore(1),
    )

    assert result == "f.txt\n\nchunk text"


async def test_generate_one_truncates_long_document_in_prompt(mocker: MockerFixture):
    """When full_document exceeds max_doc_chars, the document inside the prompt is truncated with a note appended (TC-ctxgen-30)"""
    gen = _make_generator(max_doc_chars=10)
    mock_call = mocker.patch.object(gen, "_call_llm_sync", return_value="ctx")

    await gen._generate_one(
        "chunk", "D" * 100, "f.txt", client=None, semaphore=asyncio.Semaphore(1),
    )

    prompt = mock_call.call_args[0][0]
    assert "D" * 10 + "\n...(文件過長，已截斷)" in prompt
    assert "D" * 11 not in prompt


async def test_generate_batch_order_and_length(mocker: MockerFixture):
    """generate_batch returns a list the same length as chunks, matching order one-to-one (TC-ctxgen-31)"""
    gen = _make_generator()
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    mocker.patch.object(gen, "_call_llm_sync", return_value="CTX")

    chunks = ["chunk one", "chunk two", "chunk three"]
    results = await gen.generate_batch(chunks, "full document", "file.txt")

    assert results == [f"CTX\n\n{c}" for c in chunks]
