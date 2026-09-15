
"""Unit tests for the formatting pure functions in src/fastmcp_tools/agentic_tools.py

No dependency on the DB / vLLM / any external service — tests only the four pure
functions _truncate / _visual_width / _center_line / _format_search_body.
(agentic_tools has a heavy top-level import (llama_index family) but touches no
DB / network, so it is safe to import.)
"""

import pytest

from src.fastmcp_tools.agentic_tools import (
    _BOX_WIDTH,
    _center_line,
    _format_search_body,
    _truncate,
    _visual_width,
)


def test_truncate_none_returns_empty_string():
    """A None input returns an empty string (TC-mcp-format-01)."""
    assert _truncate(None, 10) == ""


def test_truncate_short_and_exact_length_unchanged():
    """A string of length <= max_len is returned unchanged (including exactly equal) (TC-mcp-format-02)."""
    assert _truncate("abc", 10) == "abc"
    assert _truncate("abcdefghij", 10) == "abcdefghij"


def test_truncate_long_string_appends_ellipsis():
    """An over-long string is truncated to max_len-1 characters plus an ellipsis, total length == max_len (TC-mcp-format-03)."""
    out = _truncate("abcdefghijk", 10)
    assert out == "abcdefghi…"
    assert len(out) == 10


def test_truncate_coerces_non_string_input():
    """A non-string input is str()-coerced before processing (TC-mcp-format-04)."""
    assert _truncate(12345, 10) == "12345"
    assert _truncate(12345678901, 10) == "123456789…"


def test_visual_width_ascii_one_column_each():
    """Each ASCII character counts as 1 column (TC-mcp-format-05)."""
    assert _visual_width("abc 123") == 7


def test_visual_width_cjk_and_fullwidth_two_columns():
    """Each CJK and full-width character counts as 2 columns (TC-mcp-format-06)."""
    assert _visual_width("中文") == 4
    assert _visual_width("Ａ") == 2  # fullwidth A


def test_visual_width_emoji_two_columns():
    """Each emoji (0x1F000-0x1FFFF / 0x2600-0x27FF) counts as 2 columns (TC-mcp-format-07)."""
    assert _visual_width("😀") == 2
    assert _visual_width("☀") == 2


def test_visual_width_mixed_and_empty():
    """A mixed string's width is the per-character sum; an empty string is 0 (TC-mcp-format-08)."""
    assert _visual_width("") == 0
    assert _visual_width("a中b") == 4  # 1 + 2 + 1


def test_center_line_pads_ascii():
    """Centering ASCII text: pad (width - vw) // 2 spaces on the left, none on the right (TC-mcp-format-09)."""
    assert _center_line("ab", width=10) == "    ab"


def test_center_line_accounts_for_cjk_width():
    """Chinese width is computed at 2 columns before centering (TC-mcp-format-10)."""
    # "中文" has visual width 4, (10-4)//2 = 3 leading spaces
    assert _center_line("中文", width=10) == "   中文"


def test_center_line_wide_text_returned_as_is():
    """When visual width >= width, it is returned unchanged with no padding (TC-mcp-format-11)."""
    text = "x" * 12
    assert _center_line(text, width=10) == text


def test_center_line_default_width_is_box_width():
    """When width is unspecified, centering uses the _BOX_WIDTH constant (TC-mcp-format-12)."""
    out = _center_line("hi")
    assert out == " " * ((_BOX_WIDTH - 2) // 2) + "hi"


def test_format_search_body_empty_response():
    """An empty response outputs only the Query and Results lines, no Roles/Files/Scores/Hint (TC-mcp-format-13)."""
    lines = _format_search_body({})
    assert lines == ['  Query     : ""', "  Results   : 0"]


def test_format_search_body_full_response():
    """A full response outputs the roles distribution (including merged), files distribution, scores stats, and hint (TC-mcp-format-14)."""
    response = {
        "query": "營收表現",
        "total_results": 3,
        "results": [
            {"node_role": "leaf", "file_name": "a.pdf", "score": 0.9},
            {"node_role": "parent", "file_name": "a.pdf", "score": 0.8, "merged_from_leaves": 2},
            {"node_role": "expanded", "file_name": "b.txt", "score": 0.7},
        ],
        "_hint": "try a narrower query",
    }
    lines = _format_search_body(response)
    joined = "\n".join(lines)

    assert '  Query     : "營收表現"' in lines
    assert "  Results   : 3" in lines
    assert "leaf:1" in joined
    assert "parent:1(+2 merged)" in joined
    assert "expanded:1" in joined
    assert "a.pdf:2" in joined
    assert "b.txt:1" in joined
    assert "min=0.700  max=0.900  avg=0.800" in joined
    assert "  Hint      : try a narrower query" in lines


def test_format_search_body_more_than_three_files_summarized():
    """With more than 3 files, only the top 3 are listed and the rest summarized as +N more (TC-mcp-format-15)."""
    response = {
        "query": "q",
        "total_results": 4,
        "results": [
            {"node_role": "leaf", "file_name": f"f{i}.txt", "score": 0.5} for i in range(4)
        ],
    }
    joined = "\n".join(_format_search_body(response))
    assert "+1 more" in joined


def test_format_search_body_long_query_truncated():
    """An over-long query is truncated to within 70 characters plus an ellipsis (TC-mcp-format-16)."""
    response = {"query": "q" * 100, "total_results": 0, "results": []}
    lines = _format_search_body(response)
    assert lines[0] == '  Query     : "' + "q" * 69 + '…"'
