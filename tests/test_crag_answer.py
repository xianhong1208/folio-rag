"""Console CRAG answers: strict grading, original citation numbers, citation
validation, answer language, and the same guarantees on the streaming path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.domain.rag import answer as crag


def _msg(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                    finish_reason=finish)])


def _results(n):
    return [{"text": f"chunk {i + 1}", "score": 0.9 - i / 10,
             "metadata": {"file_name": f"f{i + 1}.pdf", "headings": ["Ch 1"]}} for i in range(n)]


class TestGrading:
    def test_strict_booleans(self):
        assert crag.parse_grades("[true, false]", 2) == crag.Grading([True, False])
        # The old parser did bool("false") == True.
        assert crag.parse_grades('["false", "TRUE"]', 2).grades == [False, True]

    @pytest.mark.parametrize("content, reason", [
        ("no json here", "invalid_response"),
        ("[true, oops]", "invalid_response"),
        ("[true]", "length_mismatch"),
        ('["yes", 1]', "non_boolean"),
        (None, "invalid_response"),
    ])
    def test_failures_are_classified_and_fail_open(self, content, reason):
        g = crag.parse_grades(content, 2)
        assert not g.ok and g.reason == reason and g.grades == [True, True]

    def test_llm_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("timeout")
        g = crag.grade(client, "m", "q", ["a"])
        assert g.reason == "llm_error" and crag.confidence(g, 1) == "unverified"
        assert crag.grade(client, "m", "q", []).grades == []

    def test_confidence_levels(self):
        ok = crag.Grading([True])
        assert [crag.confidence(ok, n) for n in (0, 1, 2)] == ["low", "medium", "high"]


class TestCleanup:
    def test_fullwidth_and_bold(self):
        out = crag.clean_answer("The **answer** is 42【2†L1-L3】 and __more__【3】.", [2, 3])
        assert out.text == "The answer is 42[2] and more[3]."
        assert out.citations == [2, 3] and out.invalid_citations == []

    def test_grouped_and_invalid_citations(self):
        out = crag.clean_answer("A [1, 3]. B [2]. C [9]. D [1，3]. E [x].", [1, 3])
        assert out.text == "A [1][3]. B. C. D [1][3]. E [x]."
        assert out.citations == [1, 3] and out.invalid_citations == [2, 9]

    def test_empty(self):
        assert crag.clean_answer(None, [1]).text == ""


class TestAnswer:
    def test_keeps_original_numbers_and_source_labels(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _msg("[false, true, true]"),
            _msg("Per [2] and [3] — not [1]."),
        ]
        out = crag.answer(client, "m", "¿Cuál es la política?", _results(3))
        prompt = client.chat.completions.create.call_args_list[1].kwargs["messages"][0]["content"]
        assert "[2] (f2.pdf › Ch 1) chunk 2" in prompt and "[1]" not in prompt.split("Context:")[1]
        assert "same language as the question" in prompt
        assert out["answer"] == "Per [2] and [3] — not."
        assert out["citations"] == [2, 3] and out["invalid_citations"] == [1]
        assert out["confidence"] == "high" and out["kept"] == 2 and out["dropped"] == 1

    def test_grading_failure_is_unverified_not_high(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [_msg("sure, all good"), _msg("Yes [1][2].")]
        out = crag.answer(client, "m", "q", _results(2))
        assert out["confidence"] == "unverified" and out["grading_error"] == "invalid_response"
        assert out["kept"] == 2 and "unverified" in out["note"]

    def test_gate_and_empty(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [_msg("[false]")]
        out = crag.answer(client, "m", "q", _results(1))
        assert out["answer"] == crag.NO_BASIS_MESSAGE and out["confidence"] == "low"
        client.chat.completions.create.side_effect = [_msg("[true]"), _msg("  ", "length")]
        out = crag.answer(client, "m", "q", _results(1))
        assert out["answer"] == crag.EMPTY_ANSWER_MESSAGE and "finish=length" in out["note"]
        assert crag.answer(client, "m", "q", [])["confidence"] == "none"

    def test_source_label(self):
        assert crag.source_label({"metadata": {"file_name": "a.pdf", "headings": ["H1", "H2"]}}) == "a.pdf › H1 › H2"
        assert crag.source_label({}) == ""


# ------------------------------------------------------------- streaming

def _cfg():
    return SimpleNamespace(rag=SimpleNamespace(llm=SimpleNamespace(
        provider="vllm", model="m", base_url="http://x", api_key=None, azure_deployment=None)))


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


async def _frames(results, grade_reply, pieces, gen_error=None):
    from src.api.router import admin_manage as am
    client = MagicMock()

    def create(**kw):
        if kw.get("stream"):
            if gen_error:
                raise gen_error
            return iter([_chunk(p) for p in pieces] + [SimpleNamespace(choices=[])])
        if isinstance(grade_reply, Exception):
            raise grade_reply
        return _msg(grade_reply)

    client.chat.completions.create.side_effect = create
    with patch("src.config.config_manager.Config.get_config_model", return_value=_cfg()), \
         patch("src.domain.rag.context_generator.ContextGenerator._create_sync_client",
               return_value=client):
        out = [json.loads(f[len("data: "):]) async for f in am._stream_answer_events("q", results)]
    return out, client


class TestStreaming:
    async def test_done_frame_carries_the_cleaned_validated_answer(self):
        frames, client = await _frames(_results(3), "[false, true, true]",
                                       ["See **this**【2†L1", "】 and [1, 3]", "."])
        meta, done = frames[0], frames[-1]
        assert meta["type"] == "meta" and meta["kept"] == 2 and meta["confidence"] == "high"
        assert done["type"] == "done" and done["answer"] == "See this[2] and [3]."
        assert done["citations"] == [2, 3] and done["invalid_citations"] == [1]
        prompt = [c for c in client.chat.completions.create.call_args_list
                  if c.kwargs.get("stream")][0].kwargs["messages"][0]["content"]
        assert "[3] (f3.pdf › Ch 1)" in prompt

    async def test_empty_generation_gets_a_message(self):
        frames, _ = await _frames(_results(1), "[true]", [])
        assert frames[-1]["answer"] == crag.EMPTY_ANSWER_MESSAGE

    async def test_grading_failure_is_unverified(self):
        frames, _ = await _frames(_results(2), RuntimeError("down"), ["ok [1]"])
        assert frames[0]["confidence"] == "unverified" and frames[0]["grading_error"] == "llm_error"
        assert frames[-1]["confidence"] == "unverified"

    async def test_gate(self):
        frames, _ = await _frames(_results(1), "[false]", [])
        assert frames[-1]["answer"] == crag.NO_BASIS_MESSAGE

    async def test_generation_error_frame(self):
        frames, _ = await _frames(_results(1), "[true]", [], gen_error=RuntimeError("boom"))
        assert frames[-1] == {"type": "error", "error": "boom"}

    async def test_no_results(self):
        frames, _ = await _frames([], "[]", [])
        assert frames[0]["confidence"] == "none" and frames[-1] == {"type": "done", "answer": None}


def test_eval_question_prompt_is_language_neutral():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "rag_eval.py")
    spec = importlib.util.spec_from_file_location("rag_eval_lang", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    client = MagicMock()
    client.chat.completions.create.return_value = _msg('"What is the refund window?"')
    assert m._gen_question(client, "m", "Refunds within 30 days.") == "What is the refund window?"
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "same language as the excerpt" in prompt
    assert not any("一" <= ch <= "鿿" for ch in prompt.split("Excerpt:")[0])
