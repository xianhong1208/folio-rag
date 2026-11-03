"""Golden-set retrieval eval: span matching, ranking metrics, response metrics and
deterministic sampling (evals/retrieval_metrics.py), plus the pure helpers of the
golden-set builder and the harness (scripts/eval_build_golden.py, eval_retrieval.py)."""

import importlib.util
from pathlib import Path

import pytest

from evals import retrieval_metrics as rm

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------- normalise


class TestNormalize:
    def test_strips_whitespace_punctuation_and_markup(self):
        assert rm.normalize("**Rated power** | 1500 W") == "ratedpower1500w"

    def test_full_width_and_line_breaks(self):
        assert rm.normalize("雇主應 給付\n工資１２３，") == "雇主應給付工資123"

    def test_table_triplets_match_markdown_rows(self):
        assert rm.normalize("Aster K2, Price = 3,980.") == rm.normalize("| Aster K2 | Price | 3,980 |")

    def test_text_lang(self):
        assert rm.text_lang("勞工請假規則第 4 條") == "zh"
        assert rm.text_lang("How long is the warranty?") == "en"
        assert rm.text_lang("") == "en"


# --------------------------------------------------------------- span matching


class TestSpanCoverage:
    def test_exact_containment(self):
        assert rm.span_coverage("一年內合計不得超過三十日", "一、未住院者，一年內合計不得超過三十日。") == 1.0

    def test_formatting_differences_ignored(self):
        span = "延長工作時間在二小時以內者，按平日每小時工資額加給三分之一以上"
        text = "一、延長工作時間在二小時以內者，\n按平日每小時\n工資額加給三分之一以上。"
        assert rm.span_coverage(span, text) == 1.0

    def test_partial_span_scores_fraction(self):
        span = "The House of Representatives shall chuse their Speaker and other Officers"
        text = "The House of Representatives shall chuse their Speaker"  # chunk cut mid-sentence
        cov = rm.span_coverage(span, text)
        assert 0.5 < cov < 1.0

    def test_short_span_needs_exact_match(self):
        assert rm.span_coverage("3,980", "List price 3980") == 1.0
        assert rm.span_coverage("3,980", "List price 3,890") == 0.0

    def test_unrelated_text_scores_zero(self):
        assert rm.span_coverage("勞工結婚者給予婚假十四日", "Congress shall make no law") == 0.0

    def test_empty_span(self):
        assert rm.span_coverage("", "anything") == 0.0

    def test_covers_spans_requires_all(self):
        text = "| AK2-110 | Aster K2 Smart Kettle | 3,980 | 2 years |"
        assert rm.covers_spans(["AK2-110", "3,980"], text)
        assert not rm.covers_spans(["AK2-110", "7,600"], text)
        assert not rm.covers_spans([], text)


class TestResultHits:
    EV = [{"file": "a.md", "spans": ["勞工結婚者給予婚假十四日"]},
          {"file": "b.md", "spans": ["one gallon per person per day"]}]

    def test_text_match_in_right_file(self):
        r = {"text": "第 2 條 勞工結婚者給予婚假十四日，工資照給。", "file_name": "a.md"}
        assert rm.result_hits(r, self.EV) == [0]

    def test_same_text_in_other_file_does_not_count(self):
        r = {"text": "勞工結婚者給予婚假十四日", "file_name": "c.md"}
        assert rm.result_hits(r, self.EV) == []

    def test_parent_section_containing_span_counts(self):
        parent = {"text": "Water (one gallon per person per day for several days) ... Food ...",
                  "file_name": "b.md", "chunk_range": (0, 9)}
        assert rm.result_hits(parent, self.EV) == [1]

    def test_neighbour_chunk_counts(self):
        r = {"text": "unrelated neighbour text", "file_name": "a.md", "chunk_range": (5, 5)}
        assert rm.result_hits(r, self.EV, gold_index={0: (4, 4)}) == [0]
        assert rm.result_hits(r, self.EV, gold_index={0: (4, 4)}, neighbours=0) == []
        far = dict(r, chunk_range=(7, 7))
        assert rm.result_hits(far, self.EV, gold_index={0: (4, 4)}) == []

    def test_neighbour_rule_needs_same_file(self):
        r = {"text": "x", "file_name": "b.md", "chunk_range": (5, 5)}
        assert rm.result_hits(r, self.EV, gold_index={0: (5, 5)}) == []


# ------------------------------------------------------------------- metrics


class TestRankingMetrics:
    def test_hit_at_rank_one(self):
        m = rm.ranking_metrics([[0], [], []], n_items=1)
        assert m["recall@5"] == 1.0 and m["mrr@10"] == 1.0 and m["ndcg@10"] == pytest.approx(1.0)

    def test_hit_at_rank_three(self):
        m = rm.ranking_metrics([[], [], [0]], n_items=1)
        assert m["mrr@10"] == pytest.approx(1 / 3)
        assert m["ndcg@10"] == pytest.approx(1 / 2)  # 1/log2(4)

    def test_miss_beyond_k(self):
        hits = [[] for _ in range(12)] + [[0]]
        m = rm.ranking_metrics(hits, n_items=1)
        assert m["recall@10"] == 0.0 and m["recall@20"] == 1.0 and m["mrr@10"] == 0.0

    def test_multi_hop_partial_recall(self):
        m = rm.ranking_metrics([[0], [], [], [], [], [], [1]], n_items=2)
        assert m["recall@5"] == 0.5 and m["recall@10"] == 1.0

    def test_duplicates_earn_no_gain(self):
        dup = rm.ranking_metrics([[0], [0]], n_items=2)["ndcg@10"]
        one = rm.ranking_metrics([[0], []], n_items=2)["ndcg@10"]
        assert dup == pytest.approx(one)

    def test_ideal_multi_hop(self):
        assert rm.ranking_metrics([[0], [1]], n_items=2)["ndcg@10"] == pytest.approx(1.0)

    def test_no_items(self):
        assert rm.ranking_metrics([[0]], n_items=0) == {}

    def test_file_recall(self):
        assert rm.file_recall(["a", "b", "c"], ["c"], 5) == 1.0
        assert rm.file_recall(["a", "b", "c"], ["c"], 2) == 0.0
        assert rm.file_recall(["a"], ["a", "z"], 5) == 0.5
        assert rm.file_recall(["a"], [], 5) == 0.0

    def test_first_rank(self):
        assert rm.first_rank([[], [1], [0]]) == 2
        assert rm.first_rank([[], []]) is None


class TestResponseMetrics:
    def test_estimate_tokens(self):
        assert rm.estimate_tokens("工資") == pytest.approx(1.4)
        assert rm.estimate_tokens("abcd") == pytest.approx(1.0)
        assert rm.estimate_tokens("工資ab") == pytest.approx(1.9)

    def test_duplicate_ratio(self):
        a = "The right of the people to be secure in their persons"
        assert rm.duplicate_ratio([a]) == 0.0
        assert rm.duplicate_ratio([a, a]) == pytest.approx(0.5)
        assert rm.duplicate_ratio(["short"]) == 0.0
        assert rm.duplicate_ratio([]) == 0.0

    def test_percentile(self):
        assert rm.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
        assert rm.percentile([5], 95) == 5.0
        assert rm.percentile([], 50) is None
        assert rm.percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)

    def test_mean_and_group_means(self):
        assert rm.mean([1, None, 3]) == 2
        assert rm.mean([]) is None
        rows = [{"t": "a", "x": 1.0}, {"t": "a", "x": 0.0}, {"t": "b", "x": 1.0}, {"t": "b"}]
        g = rm.group_means(rows, lambda r: r["t"], ["x"])
        assert g == {"a": {"n": 2, "x": 0.5}, "b": {"n": 2, "x": 1.0}}


# ------------------------------------------------------------------ sampling


class TestStratifiedSample:
    POOL = [{"id": f"{d}#{i}", "doc": d} for d in "abc" for i in range(5)]

    def test_deterministic_for_seed(self):
        s1 = rm.stratified_sample(self.POOL, 7, 42, stratum=lambda p: p["doc"])
        s2 = rm.stratified_sample(list(reversed(self.POOL)), 7, 42, stratum=lambda p: p["doc"])
        assert [p["id"] for p in s1] == [p["id"] for p in s2]

    def test_seed_changes_selection(self):
        s1 = rm.stratified_sample(self.POOL, 7, 1, stratum=lambda p: p["doc"])
        s2 = rm.stratified_sample(self.POOL, 7, 2, stratum=lambda p: p["doc"])
        assert [p["id"] for p in s1] != [p["id"] for p in s2]

    def test_spread_over_strata(self):
        s = rm.stratified_sample(self.POOL, 6, 7, stratum=lambda p: p["doc"])
        counts = {d: sum(1 for p in s if p["doc"] == d) for d in "abc"}
        assert counts == {"a": 2, "b": 2, "c": 2}

    def test_exclusion_and_exhaustion(self):
        s = rm.stratified_sample(self.POOL, 100, 3, stratum=lambda p: p["doc"],
                                 exclude={"a#0", "b#1"})
        ids = [p["id"] for p in s]
        assert len(ids) == 13 and "a#0" not in ids and len(set(ids)) == 13


# --------------------------------------------------------- builder / harness


@pytest.fixture(scope="module")
def builder():
    return _load("eval_build_golden")


@pytest.fixture(scope="module")
def harness():
    return _load("eval_retrieval")


class TestBuilderHelpers:
    DOC = ("# Test Law\n\nSource: somewhere.\n\n## 第 一 章\n\n### 第 1 條\n"
           "勞工結婚者給予**婚假**十四日，工資照給。勞工喪假依左列規定辦理之。\n\n"
           "## Prices\n\n| SKU | Price |\n|---|---|\n| AK2 | 3,980 |\n| AK1 | 1,890 |\n")

    def test_find_exact_verbatim_and_markup_insensitive(self, builder):
        text = "勞工結婚者給予**婚假**十四日，\n工資照給。"
        assert builder.find_exact("給予婚假十四日", text) == "給予**婚假**十四日"
        assert builder.find_exact("十四日，工資照給", text) == "十四日，\n工資照給"
        assert builder.find_exact("婚假三十日", text) is None
        assert builder.find_exact("", text) is None

    def test_split_passages(self, builder):
        ps = builder.split_passages("law.md", self.DOC, min_norm=10)
        prose = [p for p in ps if p["kind"] == "prose"]
        rows = [p for p in ps if p["kind"] == "table_row"]
        assert [p["heading"] for p in prose] == ["第 1 條"]
        assert prose[0]["lang"] == "zh" and "Source:" not in prose[0]["text"]
        assert [r["row"] for r in rows] == ["| AK2 | 3,980 |", "| AK1 | 1,890 |"]
        assert rows[0]["text"].splitlines()[1] == "| SKU | Price |"
        assert len({p["id"] for p in ps}) == len(ps)

    def test_programmatic_check(self, builder):
        chk = builder.programmatic_check
        assert chk("zh_factual", "zh", "勞工結婚可以請幾天婚假？", []) is None
        assert chk("zh_factual", "zh", "How many days of marriage leave?", []) is not None
        assert chk("zh_factual", "zh", "劳工结婚可以请几天婚假？", []) == \
            "question contains Simplified characters"
        assert chk("en_factual", "en", "What does the passage say about leave?", []) is not None
        assert chk("keyword", "en", "marriage leave days", []) is None
        assert chk("keyword", "en", "how many days of marriage leave do workers get", []) is not None
        assert chk("keyword", "zh", "婚假 天數", []) is None
        assert chk("paraphrase", "en", "How many days off for a wedding?",
                   ["marriage leave of fourteen days"]) is None
        assert chk("paraphrase", "en", "marriage leave of fourteen days?",
                   ["marriage leave of fourteen days"]) is not None

    def test_word_overlap(self, builder):
        assert builder.word_overlap("婚假天數", "婚假十四日") > 0
        assert builder.word_overlap("What is the warranty?", "warranty is 2 years") == 1.0


class TestHarnessHelpers:
    def test_score_results(self, harness):
        q = {"evidence": [{"file": "a.md", "spans": ["勞工結婚者給予婚假十四日"]}]}
        res = [{"text": "nothing", "file_name": "b.md", "score": 0.9},
               {"text": "勞工結婚者給予婚假十四日，工資照給。", "file_name": "a.md", "score": 0.6}]
        s = harness.score_results(res, q, {})
        assert s["first_rank"] == 2 and s["recall@5"] == 1.0 and s["mrr@10"] == 0.5
        assert s["file_recall@5"] == 1.0 and s["top_score"] == 0.9

    def test_score_unanswerable(self, harness):
        s = harness.score_results([], {"evidence": []}, {})
        assert s["top_score"] is None and "recall@5" not in s

    def test_mcp_and_trace_results(self, harness):
        data = {"results": [{"text": "t", "file_name": "a.md", "score": 0.5,
                             "node_role": "parent", "chunk_index_range": [2, 6]}]}
        assert harness.mcp_results(data)[0]["chunk_range"] == (2, 6)
        idx = {("a.md", "leaf"): 3}
        tr = {"trace": {"results": [{"text": "leaf", "final_score": 0.7, "vector_rank": 4,
                                     "metadata": {"file_name": "a.md"}}]}}
        r = harness.trace_results(tr, idx)[0]
        assert r["chunk_range"] == (3, 3) and r["vector_rank"] == 4

    def test_summarise_and_variance(self, harness):
        def row(i, typ, lang, rank, score, ans=True):
            s = {"first_rank": rank, "top_score": score}
            if ans:
                s.update({"recall@5": 1.0 if rank and rank <= 5 else 0.0, "recall@10": 1.0,
                          "recall@20": 1.0, "mrr@10": 1.0 / rank if rank else 0.0,
                          "ndcg@10": 1.0, "file_recall@5": 1.0})
            mcp = dict(s, ms=100.0 + i, tokens=500.0, dup_ratio=0.1, roles=["leaf"], n_results=3)
            tr = dict(s, ms=50.0, stage_ranks={"vector_rank": 2})
            return {"id": f"q{i}", "type": typ, "lang": lang, "answerable": ans,
                    "mcp": mcp, "mcp_k20": dict(s), "trace": tr}
        rows = [row(1, "zh_factual", "zh", 1, 0.9), row(2, "keyword", "en", 7, 0.4),
                row(3, "unanswerable", "zh", None, 0.6, ans=False)]
        s = harness.summarise(rows)
        assert s["n_answerable"] == 2
        assert s["mcp"]["overall"]["recall@5"] == 0.5
        assert s["mcp"]["thresholds"]["false_accept@0.5"] == 1.0
        assert s["mcp"]["thresholds"]["answerable_accept@0.5"] == 0.5
        assert set(s["mcp"]["by_type"]) == {"zh_factual", "keyword"}
        assert s["trace_stages"]["dense_in_top10_lb"] == 1.0
        v = harness.variance([s, s])
        assert v["mcp.recall@5"]["max_abs_diff"] == 0
        assert harness.rank_flips([rows, rows]) == {"mcp": [], "mcp_k20": [], "trace": []}
