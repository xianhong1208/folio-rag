"""docling conversion guard: partial results retried once and reported, readable failures."""

from types import SimpleNamespace

import pytest

from src.domain.rag import docling_guard as dg


def _err(page=None, msg="boom", module="layout"):
    return SimpleNamespace(page_no=page, error_message=msg, module_name=module)


def _res(status, errors=()):
    return SimpleNamespace(status=SimpleNamespace(value=status), errors=list(errors))


class _Seq:
    """convert() stub: returns/raises the queued outcomes and counts the calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _run(seq):
    return dg.convert_checked(seq, "f.docx")


def test_success_passes_through():
    ok = _res("success")
    seq = _Seq(ok)
    assert _run(seq) == (ok, None) and seq.calls == 1


def test_raised_error_is_not_retried():
    seq = _Seq(ValueError("broken zip"))
    with pytest.raises(ValueError, match="broken zip"):
        _run(seq)
    assert seq.calls == 1


def test_partial_retried_once_and_best_kept_with_page_warning():
    seq = _Seq(
        _res("partial_success", [_err(4), _err(7)]),
        _res("partial_success", [_err(7, "properties does not have key 'FontBBox'")]),
    )
    res, warning = _run(seq)
    assert dg.missing_pages(res) == [7] and seq.calls == 2
    assert warning.startswith("Partial parse: page 7 could not be parsed and is missing")


def test_first_partial_kept_when_the_retry_is_worse():
    first = _res("partial_success", [_err(2)])
    seq = _Seq(first, _res("partial_success", [_err(2), _err(3)]))
    res, warning = _run(seq)
    assert res is first and "page 2" in warning


def test_partial_then_success_on_retry():
    seq = _Seq(_res("partial_success", [_err(1)]), _res("success"))
    assert _run(seq)[1] is None


def test_failure_then_success_on_retry():
    ok = _res("success")
    assert _run(_Seq(_res("failure", [_err(None)]), ok)) == (ok, None)


def test_failure_raises_readable_value_error():
    seq = _Seq(_res("failure", [_err(None, "no pages")]), _res("failure", [_err(None, "no pages")]))
    with pytest.raises(ValueError, match="could not parse 'f.docx': layout: no pages"):
        _run(seq)


def test_failure_after_partial_keeps_partial():
    seq = _Seq(_res("partial_success", [_err(2)]), _res("failure", [_err(None)]))
    res, warning = _run(seq)
    assert dg.missing_pages(res) == [2] and "page 2" in warning


def test_partial_without_page_numbers_gives_generic_warning():
    seq = _Seq(_res("partial_success", [_err(None)]), _res("partial_success", [_err(None)]))
    _, warning = _run(seq)
    assert "part of the document could not be parsed" in warning


def test_partial_warning_plural_and_none():
    assert "pages 2, 5 could not be parsed and are missing" in dg.partial_warning([2, 5])
    assert dg.partial_warning([]) is None


def test_no_error_detail():
    seq = _Seq(_res("failure"), _res("failure"))
    with pytest.raises(ValueError, match="no error detail"):
        _run(seq)
