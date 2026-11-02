"""Contextual prefix scope, language and script hygiene (Simplified drift, long-document bug)."""

import re

import pytest
from pytest_mock import MockerFixture

from src.domain.rag.context_generator import ContextGenerator, _CONTEXT_PROMPT
from src.domain.rag import context_scope as cs
from src.domain.rag.han_script import (
    document_language,
    fix_simplified,
    is_traditional,
    strip_invisible,
)


def _gen(**kw):
    return ContextGenerator(llm_config=None, model="m", **kw)


async def test_chunk_past_60k_chars_sees_its_own_neighbourhood(mocker: MockerFixture):
    # Latent bug: the prompt carried full_document[:60000], so a chunk beyond that
    # point was described from an unrelated beginning of the document.
    filler = "Routine paragraph about site access and paperwork. " * 1400  # ~72k chars
    late = "The K7 pressure gauge is calibrated every 45 days."
    doc = filler + "Chapter 29 covers gauge calibration. " + late + " End of chapter."
    gen = _gen(max_doc_chars=60000)
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    call = mocker.patch.object(gen, "_call_llm_sync", return_value="Gauge calibration rules")
    await gen.generate_prefixes([late], full_document=doc, file_name="handbook.md")
    prompt = call.call_args[0][0]
    assert "Chapter 29 covers gauge calibration." in prompt
    assert len(prompt) < 20_000


def test_prompt_is_language_neutral():
    # No hard-coded Chinese instructions or examples: the language follows the document.
    instructions = _CONTEXT_PROMPT.split("{file_name}")[0]
    assert not re.search(r"[一-鿿]", instructions)


async def test_language_hint_and_scope_in_prompt(mocker: MockerFixture):
    gen = _gen()
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    call = mocker.patch.object(gen, "_call_llm_sync", return_value="說明保固期間")
    await gen.generate_prefixes(
        ["保固期間為三十六個月。"], full_document="本文件說明保固與客服。" * 10, file_name="a.md",
        contexts=["## 四、保固與客服\n保固期間為三十六個月。"], outline="- 方案說明 › 四、保固與客服",
        language="Traditional Chinese")
    prompt = call.call_args[0][0]
    assert "Traditional Chinese" in prompt
    assert "四、保固與客服" in prompt and "<outline>" in prompt


async def test_simplified_drift_and_zero_width_are_cleaned(mocker: MockerFixture):
    gen = _gen()
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    mocker.patch.object(gen, "_call_llm_sync", return_value="伺服器部署文档章节​，说明ROC​m安装")
    doc = "伺服器部署文件章節,說明 ROCm 驅動程式的安裝與設定步驟。" * 5
    out = await gen.generate_prefixes(["說明 ROCm 安裝"], full_document=doc, file_name="d.md")
    assert out == ["伺服器部署文檔章節，說明ROCm安裝"]


async def test_english_prefix_untouched(mocker: MockerFixture):
    gen = _gen()
    mocker.patch.object(ContextGenerator, "_create_sync_client", return_value=None)
    mocker.patch.object(gen, "_call_llm_sync", return_value="Warranty terms of the Aurora X2 router")
    out = await gen.generate_prefixes(["36-month warranty."], full_document="Aurora manual. " * 20,
                                      file_name="a.docx")
    assert out == ["Warranty terms of the Aurora X2 router"]


def test_cache_key_tracks_prompt_version():
    from src.domain.rag.context_generator import PROMPT_VERSION
    assert f"prompt={PROMPT_VERSION}" in _gen().cache_key
    assert PROMPT_VERSION != "1"  # bumped: section scope + language-neutral prompt


# --- context_scope ---------------------------------------------------------------

def test_outline_unique_ordered_and_capped():
    out = cs.document_outline([["M", "1"], ["M", "1"], ["M", "2"], None])
    assert out == "- M › 1\n- M › 2"
    long = cs.document_outline([["H" + str(i) * 50] for i in range(200)], max_chars=300)
    assert len(long) <= 310 and long.endswith("- …")


def test_section_texts_group_by_section_and_window():
    raws = ["a" * 10, "b" * 10, "c" * 10]
    heads = [["M", "1"], ["M", "1", "1.1"], ["M", "2"]]
    out = cs.section_texts(raws, heads, width=1000)
    assert out[0] == out[1] == "a" * 10 + "\n\n" + "b" * 10
    assert out[2] == "c" * 10
    narrow = cs.section_texts(["x" * 100, "TARGET", "y" * 100], [None] * 3, width=40)
    assert "TARGET" in narrow[1] and narrow[1].startswith("…") and narrow[1].endswith("…")


def test_window_and_locate():
    assert cs.window("short", 0, 5, 100) == "short"
    text = "0123456789" * 10
    w = cs.window(text, 0, 2, 20)
    assert w == text[:20] + "…"
    assert "NEEDLE" in cs.locate_window("x" * 500 + "NEEDLE" + "y" * 500, "NEEDLE", 50)
    assert cs.locate_window("abc", "zzz", 2).startswith("ab")


# --- han_script -----------------------------------------------------------------

def test_strip_invisible():
    assert strip_invisible("ROC​m x y﻿") == "ROCm x y"
    assert strip_invisible("👨‍👩") == "👨‍👩"  # ZWJ kept for emoji
    assert strip_invisible("") == ""


def test_fix_simplified_only_for_traditional_documents():
    trad_doc = "本文件說明晨曦雲端儲存的資料保護與計費方式,台灣客服專線。" * 3
    assert fix_simplified("说明文件与软件", trad_doc) == "說明文件與軟件"  # 文件 kept, not 檔案
    simp_doc = "本文件说明晨曦云端储存的资料保护与计费方式。" * 3
    assert fix_simplified("说明文件", simp_doc) == "说明文件"
    assert fix_simplified("台灣的服務", trad_doc) == "台灣的服務"  # characters the document uses stay
    assert fix_simplified("English only", trad_doc) == "English only"


@pytest.mark.parametrize("text,lang", [
    ("本文件說明保固與客服的服務內容。" * 3, "Traditional Chinese"),
    ("本文件说明保固与客服的服务内容。" * 3, "Simplified Chinese"),
    ("これは日本語の文書です。カタカナもあります。" * 3, "Japanese"),
    ("이 문서는 한국어로 작성되었습니다. 제품 설명서입니다." * 3, "Korean"),
    ("An English manual about routers. " * 5, None),
])
def test_document_language(text, lang):
    assert document_language(text) == lang


def test_is_traditional_undecided_on_little_evidence():
    assert is_traditional("ABC 123") is None
