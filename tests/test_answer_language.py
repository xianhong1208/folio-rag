"""The answer is written in the question's language, and Traditional stays Traditional.

QA 2026-11-09: an English question over Chinese context got a Chinese answer — in
Simplified characters ("负责人的姓名是…" for "Who is responsible…?"). The prompt only
said "same language as the question", which the model read as the context's.
"""

import pytest

from src.domain.rag.answer import Context, answer_language, answer_prompt, finish_stream, fit_script, Grading

ZH_TW = [Context(number=1, text="本專案的負責人是林雅婷工程師，負責系統維運與資料庫備份。", source="a.pdf")]
ZH_CN = [Context(number=1, text="本项目的负责人是林雅婷工程师，负责系统运维与数据库备份。", source="a.pdf")]
EN = [Context(number=1, text="The project owner is the operations team.", source="a.pdf")]


class TestAnswerLanguage:
    @pytest.mark.parametrize("query, contexts, expected", [
        ("Who is responsible for backups?", ZH_TW, "Latin"),
        ("¿Quién es el responsable?", ZH_TW, "Latin"),
        ("誰負責資料庫備份？", ZH_TW, "Traditional Chinese"),
        ("谁负责数据库备份？", ZH_TW, "Simplified Chinese"),
        ("負責人是誰", EN, "Traditional Chinese"),
        ("负责人是谁", EN, "Simplified Chinese"),
        ("名字", ZH_CN, "Simplified Chinese"),   # no script-specific character: follow the context
        ("名字", ZH_TW, "Traditional Chinese"),
        ("バックアップの担当者は？", ZH_TW, "Japanese"),
        ("백업 담당자는 누구입니까?", ZH_TW, "Korean"),
        ("12345 ???", ZH_TW, None),
    ])
    def test_language_comes_from_the_question(self, query, contexts, expected):
        assert answer_language(query, contexts) == expected

    def test_english_question_over_chinese_context_is_told_not_to_answer_in_chinese(self):
        prompt = answer_prompt("Who is responsible for backups?", ZH_TW)
        assert "do not answer in Chinese" in prompt

    def test_traditional_question_is_told_to_avoid_simplified(self):
        prompt = answer_prompt("誰負責資料庫備份？", ZH_TW)
        assert "Traditional Chinese" in prompt and "never use Simplified" in prompt


class TestFitScript:
    def test_simplified_drift_in_a_traditional_answer_is_fixed(self):
        out = fit_script("负责人的姓名是林雅婷工程师 [1]", "誰負責資料庫備份？", ZH_TW)
        assert "負責人" in out and "负" not in out and "[1]" in out

    def test_english_answers_are_untouched(self):
        text = "The owner is 林雅婷 [1]."
        assert fit_script(text, "Who is responsible?", ZH_TW) == text

    def test_simplified_questions_keep_simplified_answers(self):
        text = "负责人是林雅婷 [1]"
        assert fit_script(text, "谁负责数据库备份？", ZH_CN) == text

    def test_a_simplified_question_over_traditional_context_keeps_simplified(self):
        text = "数据库由林雅婷维护 [1]"  # simplified characters the question itself does not use
        assert fit_script(text, "谁负责备份？", ZH_TW) == text

    def test_the_streamed_answer_is_fixed_too(self):
        done = finish_stream("负责人是林雅婷 [1]", ZH_TW, Grading(grades=[True], ok=True), "誰負責備份？")
        assert "負責人" in done["answer"]
