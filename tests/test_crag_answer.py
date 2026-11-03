"""Console CRAG answers: strict grading, original citation numbers, citation
validation, answer language, and the same guarantees on the streaming path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.domain.rag import answer as crag
from src.domain.rag.token_budget import estimate_tokens


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


class TestExcerpts:
    """Merged sections are 3-4k characters: grading and generation see the part
    around the hit leaves (metadata.hit_spans), within a token budget."""

    SECTION = "前言" * 1500 + "<HIT>年假七日</HIT>" + "附錄" * 1500

    def _merged(self):
        s = self.SECTION.index("<HIT>")
        return {"text": self.SECTION, "score": 0.9,
                "metadata": {"file_name": "p.pdf", "hit_spans": [[s, s + 15]]}}

    def test_grade_texts_centre_on_the_hit(self):
        (t,) = crag.grade_texts([self._merged()])
        assert "<HIT>年假七日</HIT>" in t
        assert estimate_tokens(t.replace("…", "")) <= crag.GRADE_TOKENS

    def test_contexts_centre_on_the_hit_within_budget(self):
        (c,) = crag.contexts_for([self._merged()], [True])
        assert "<HIT>" in c.text and estimate_tokens(c.text.replace("…", "")) <= crag.CONTEXT_TOKENS
        assert "<HIT>" in crag.answer_prompt("q", [c])

    def test_without_spans_the_head_within_budget(self):
        r = {"text": "長" * 5000, "metadata": {}}
        (t,) = crag.grade_texts([r])
        assert t == "長" * int(crag.GRADE_TOKENS / 0.7)

    def test_prompts_cap_unexcerpted_text(self):
        prompt = crag.grade_prompt("q", ["字" * 5000])
        assert prompt.count("字") <= crag.GRADE_TOKENS / 0.7 + 1

    def test_answer_flow_grades_the_excerpt(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [_msg("[true]"), _msg("Seven days [1].")]
        out = crag.answer(client, "m", "年假幾天?", [self._merged()])
        grade_prompt = client.chat.completions.create.call_args_list[0].kwargs["messages"][0]["content"]
        assert "<HIT>" in grade_prompt and out["citations"] == [1]


class TestHitSpans:
    def test_spans_locate_leaves_in_the_parent(self):
        from src.domain.rag.query_engine import hit_spans

        parent = "alpha\n\nbeta gamma\n\ndelta"
        assert hit_spans(parent, ["beta gamma", "delta"]) == [[7, 17], [19, 24]]

    def test_prefixed_leaf_found_by_its_tail_and_missing_skipped(self):
        from src.domain.rag.query_engine import hit_spans

        body = "".join(chr(0x4E00 + i) for i in range(200))
        parent = "head line here\n\n" + body + "\n\ntail"
        start = parent.index(body)                                # 16
        spans = hit_spans(parent, ["CTX: " + body, "nowhere", "", None])
        assert spans == [[start - 5, start + 200]]                # tail-anchored
        assert hit_spans("", ["a"]) == []

    def test_merged_result_carries_spans(self):
        from llama_index.core.schema import TextNode

        from src.domain.rag.merge_plan import MergeGroup
        from src.domain.rag.query_engine import _to_result
        from tests.test_hybrid_search import nws

        p = TextNode(id_="00000000-0000-0000-0000-0000000000aa", text="one\n\ntwo\n\nthree",
                     metadata={"node_role": "parent", "file_id": "f", "file_name": "a"})
        hits = [nws("00000000-0000-0000-0000-000000000001", 0.9, text="three"),
                nws("00000000-0000-0000-0000-000000000002", 0.5, text="one")]
        r = _to_result(MergeGroup(parent=p, hits=hits))
        assert r.metadata.hit_spans == [[10, 15], [0, 3]] and r.metadata.merged_from == 2
