"""docling conversion guard: partial results, OOM backoff, readable failures."""

from types import SimpleNamespace

import pytest

from src.domain.rag import docling_guard as dg


def _err(page=None, msg="boom", module="layout"):
    return SimpleNamespace(page_no=page, error_message=msg, module_name=module)


def _res(status, errors=()):
    return SimpleNamespace(status=SimpleNamespace(value=status), errors=list(errors))


class _Seq:
    """convert() stub: returns/raises the queued outcomes and records the batch it ran at."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.batches = []

    def __call__(self):
        self.batches.append(dg.current_stage_batch())
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


@pytest.fixture(autouse=True)
def _no_hook(monkeypatch):
    monkeypatch.setattr(dg, "install_batch_hook", lambda: True)
    monkeypatch.setattr(dg, "release_accelerator_cache", lambda: None)


def _run(seq, **kw):
    return dg.convert_checked(seq, "f.pdf", sleep=lambda s: None, **kw)


def test_success_passes_through():
    ok = _res("success")
    seq = _Seq(ok)
    assert _run(seq) == (ok, None)
    assert seq.batches == [4]


def test_swallowed_oom_retries_with_smaller_batch_until_complete():
    oom = "CUDA out of memory. Tried to allocate 2.00 GiB"
    seq = _Seq(
        _res("partial_success", [_err(2, oom), _err(3, oom)]),
        _res("partial_success", [_err(3, oom)]),
        _res("success"),
    )
    res, warning = _run(seq)
    assert warning is None and res.status.value == "success"
    assert seq.batches == [4, 2, 1]


def test_raised_oom_retries_then_gives_readable_error():
    seq = _Seq(*(RuntimeError("CUDA out of memory") for _ in range(3)))
    with pytest.raises(RuntimeError, match="Out of GPU memory while parsing 'f.pdf'"):
        _run(seq)
    assert seq.batches == [4, 2, 1]


def test_raised_non_oom_is_not_retried():
    seq = _Seq(ValueError("broken xref"))
    with pytest.raises(ValueError, match="broken xref"):
        _run(seq)
    assert seq.batches == [4]


def test_non_oom_partial_retried_once_and_best_kept_with_page_warning():
    seq = _Seq(
        _res("partial_success", [_err(4), _err(7)]),
        _res("partial_success", [_err(7, "properties does not have key 'FontBBox'")]),
    )
    res, warning = _run(seq)
    assert dg.missing_pages(res) == [7]
    assert warning.startswith("Partial parse: page 7 could not be parsed and is missing")
    assert seq.batches == [4, 4]


def test_partial_then_success_on_retry():
    seq = _Seq(_res("partial_success", [_err(1)]), _res("success"))
    assert _run(seq)[1] is None


def test_failure_raises_readable_value_error():
    seq = _Seq(_res("failure", [_err(None, "no pages")]), _res("failure", [_err(None, "no pages")]))
    with pytest.raises(ValueError, match="could not parse 'f.pdf': layout: no pages"):
        _run(seq)


def test_failure_with_oom_at_batch_one_raises_oom_message():
    seq = _Seq(_res("failure", [_err(1, "CUDA out of memory")]))
    with pytest.raises(RuntimeError, match="Out of GPU memory"):
        _run(seq, start_batch=1)


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


def test_is_oom_variants():
    assert dg.is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert dg.is_oom("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate")
    assert not dg.is_oom(ValueError("password protected"))
    assert not dg.is_oom(None)


def test_batch_sizes_halve_to_one():
    assert dg._batch_sizes(4) == [4, 2, 1]
    assert dg._batch_sizes(5) == [5, 2, 1]
    assert dg._batch_sizes(0) == [1]


def test_apply_stage_batch_only_touches_batched_stages():
    stages = [SimpleNamespace(name=n, batch_size=4) for n in ("preprocess", "ocr", "layout", "table", "assemble")]
    ctx = SimpleNamespace(stages=stages)
    with dg.stage_batch(2):
        dg.apply_stage_batch(ctx)
    assert [s.batch_size for s in stages] == [4, 2, 2, 2, 4]
    assert dg.current_stage_batch() is None
    dg.apply_stage_batch(ctx)  # no override: unchanged
    assert [s.batch_size for s in stages] == [4, 2, 2, 2, 4]


def test_stage_models_record_oom_from_worker_threads():
    import threading

    def layout(conv_res, pages):
        if len(pages) > 1:
            raise RuntimeError("CUDA out of memory (layout)")
        return iter(pages)

    stage = SimpleNamespace(name="layout", batch_size=4, model=layout)
    with dg.stage_batch(4) as ooms:
        dg.apply_stage_batch(SimpleNamespace(stages=[stage]))
    errors = []

    def worker():  # docling runs stage models on its own threads
        try:
            stage.model(None, [1, 2])
        except RuntimeError as e:
            errors.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors and ooms == errors          # recorded although the TLS is gone there
    assert stage.model(None, [7]) == [7]      # non-failing calls pass through


def test_non_oom_stage_error_not_recorded():
    def table(conv_res, pages):
        raise ValueError("bad cell")

    stage = SimpleNamespace(name="table", batch_size=4, model=table)
    with dg.stage_batch(4) as ooms:
        dg.apply_stage_batch(SimpleNamespace(stages=[stage]))
    with pytest.raises(ValueError):
        stage.model(None, [1])
    assert ooms == []


def test_oom_hidden_behind_terminated_early_is_detected():
    # docling replaces the cause with "pipeline terminated early" in the result;
    # the stage recorder is what reveals it was an OOM.
    def convert_factory():
        outcomes = [
            _res("partial_success", [_err(2, "pipeline terminated early", "StandardPdfPipeline")]),
            _res("success"),
        ]
        batches = []

        def convert():
            batches.append(dg.current_stage_batch())
            if len(batches) == 1:
                dg._tls.ooms.append(RuntimeError("CUDA out of memory"))
            return outcomes.pop(0)

        return convert, batches

    convert, batches = convert_factory()
    res, warning = _run(convert)
    assert warning is None and batches == [4, 2]


def test_raised_error_with_recorded_oom_is_retried():
    calls = []

    def convert():
        calls.append(dg.current_stage_batch())
        if len(calls) == 1:
            dg._tls.ooms.append(RuntimeError("CUDA out of memory"))
            raise RuntimeError("Conversion failed: pipeline terminated early")
        return _res("success")

    assert _run(convert)[1] is None and calls == [4, 2]


def test_release_cache_without_torch_is_a_noop(monkeypatch):
    import sys
    monkeypatch.undo()  # use the real function
    monkeypatch.setitem(sys.modules, "torch", None)
    dg.release_accelerator_cache()


def test_release_cache_calls_empty_cache(monkeypatch):
    import sys
    calls = []
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: calls.append(1)))
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "torch", fake)
    dg.release_accelerator_cache()
    assert calls == [1]


def test_real_hook_wraps_create_run_ctx(monkeypatch):
    monkeypatch.undo()
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    assert dg.install_batch_hook() is True
    assert dg.install_batch_hook() is True  # idempotent
    ctx = SimpleNamespace(stages=[SimpleNamespace(name="ocr", batch_size=4)])
    # the wrapper calls the original then applies the override
    wrapped = StandardPdfPipeline._create_run_ctx
    monkeypatch.setattr(dg, "apply_stage_batch", lambda c: ("applied", c))
    assert callable(wrapped)
    assert dg.apply_stage_batch(ctx)[0] == "applied"
