
"""Evaluation-metric pure functions (evals/metrics.py).

CER (character error rate), simplified-character ratio (heuristic), recall@k,
and MRR — all pure logic, shared by the scoring scripts
(scripts/run_rag_eval.py / run_asr_eval.py)."""

import pytest

from evals.metrics import cer, mrr, recall_at_k, simplified_char_ratio


class TestCER:
    def test_identical_zero(self):
        assert cer("今天開會討論預算", "今天開會討論預算") == 0.0

    def test_completely_different(self):
        assert cer("甲乙丙", "子丑寅") == 1.0

    def test_single_substitution(self):
        # 8-character reference, 1 character substituted → 1/8
        assert cer("今天開會討論預算", "今天開會討論預專") == pytest.approx(1 / 8)

    def test_insertion_and_deletion(self):
        assert cer("開會", "開個會") == pytest.approx(1 / 2)   # 1 insertion / reference length 2
        assert cer("開個會", "開會") == pytest.approx(1 / 3)   # 1 deletion / reference length 3

    def test_whitespace_and_punct_normalized(self):
        """Default normalization: whitespace and punctuation are ignored (ASR output often lacks punctuation and should not be penalized for it)."""
        assert cer("今天,開會。", "今天開會") == 0.0
        assert cer("today meeting", "todaymeeting") == 0.0

    def test_empty_reference_guard(self):
        assert cer("", "任何輸出") == 1.0
        assert cer("", "") == 0.0


class TestSimplifiedRatio:
    def test_pure_traditional(self):
        assert simplified_char_ratio("會議紀錄與檢索系統") == 0.0

    def test_pure_simplified(self):
        r = simplified_char_ratio("会议记录与检索系统")
        assert r > 0.5, f"明顯簡體文本應被偵測,實得 {r}"

    def test_no_cjk_returns_zero(self):
        assert simplified_char_ratio("hello 123") == 0.0


class TestRetrievalMetrics:
    def test_recall_at_k_hit(self):
        assert recall_at_k(retrieved=["a.pdf", "b.pdf"], expected=["b.pdf"], k=2) == 1.0

    def test_recall_at_k_miss_outside_k(self):
        assert recall_at_k(retrieved=["a", "b", "c"], expected=["c"], k=2) == 0.0

    def test_recall_partial(self):
        # Two expected files, one hit within k → 0.5
        assert recall_at_k(retrieved=["a", "x"], expected=["a", "b"], k=2) == 0.5

    def test_mrr_first_hit_position(self):
        assert mrr(retrieved=["x", "hit", "y"], expected=["hit"]) == pytest.approx(1 / 2)
        assert mrr(retrieved=["hit"], expected=["hit"]) == 1.0
        assert mrr(retrieved=["x", "y"], expected=["hit"]) == 0.0
