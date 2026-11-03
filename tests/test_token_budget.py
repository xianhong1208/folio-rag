"""Token estimates, truncation and span-centred excerpts (src/domain/rag/token_budget.py)."""

from __future__ import annotations

from src.domain.rag import token_budget as tb


class TestEstimate:
    def test_cjk_and_latin_rates(self):
        assert tb.estimate_tokens("") == 0
        assert tb.estimate_tokens("一" * 10) == 7                 # 0.7 / CJK char
        assert tb.estimate_tokens("abcd" * 10) == 10              # 0.25 / other char
        assert tb.estimate_tokens("年假 days") == 3               # 1.4 + 1.25 → ceil

    def test_kana_hangul_fullwidth_count_as_cjk(self):
        for ch in ("あ", "カ", "한", "，", "　"):
            assert tb.char_cost(ch) == tb.CJK_TOKENS_PER_CHAR, ch
        assert tb.char_cost("a") == tb.OTHER_TOKENS_PER_CHAR

    def test_prefix_costs(self):
        p = tb.cost_prefix("a一")
        assert p == [0.0, 0.25, 0.95]


class TestTruncate:
    def test_fits_unchanged(self):
        assert tb.truncate_to_tokens("short", 10) == ("short", False)

    def test_cuts_at_budget(self):
        text, cut = tb.truncate_to_tokens("一" * 100, 7)
        assert cut and text == "一" * 10
        assert tb.estimate_tokens(text) <= 7

    def test_edge_cases(self):
        assert tb.truncate_to_tokens("", 5) == ("", False)
        assert tb.truncate_to_tokens("abc", 0) == ("", True)


class TestExcerpt:
    TEXT = "A" * 400 + "<HIT>" + "B" * 400 + "<TWO>" + "C" * 400

    def test_whole_text_when_it_fits(self):
        assert tb.excerpt("small", [[0, 2]], 100) == "small"
        assert tb.excerpt("", [[0, 1]], 10) == ""

    def test_centred_on_span_within_budget(self):
        s = self.TEXT.index("<HIT>")
        out = tb.excerpt(self.TEXT, [[s, s + 5]], 50)
        assert "<HIT>" in out
        assert out.startswith("…") and out.endswith("…")
        assert tb.estimate_tokens(out.replace("…", "").strip()) <= 50
        # grown on both sides, not just the hit
        core = out.strip(" …")
        assert core.index("<HIT>") > 50 and len(core) - core.index("<HIT>") > 50

    def test_two_spans_both_kept_with_gap_marker(self):
        s1, s2 = self.TEXT.index("<HIT>"), self.TEXT.index("<TWO>")
        out = tb.excerpt(self.TEXT, [[s1, s1 + 5], [s2, s2 + 5]], 60)
        assert "<HIT>" in out and "<TWO>" in out and " … " in out

    def test_spans_in_priority_order_when_not_all_fit(self):
        s1, s2 = self.TEXT.index("<HIT>"), self.TEXT.index("<TWO>")
        out = tb.excerpt(self.TEXT, [[s2, s2 + 5], [s1 - 300, s1 + 5]], 3)
        assert "<TWO>" in out and "<HIT>" not in out

    def test_oversized_first_span_is_cut(self):
        out = tb.excerpt(self.TEXT, [[0, len(self.TEXT)]], 20)
        assert tb.estimate_tokens(out.replace("…", "").strip()) <= 20

    def test_without_usable_spans_returns_the_head(self):
        out = tb.excerpt(self.TEXT, [], 10)
        assert out == "A" * 40
        assert tb.excerpt(self.TEXT, [["x"], [5, 5], [None, 3]], 10) == "A" * 40

    def test_span_at_text_start_has_no_leading_ellipsis(self):
        out = tb.excerpt(self.TEXT, [[0, 5]], 20)
        assert out.startswith("AAAA") and out.endswith("…")
