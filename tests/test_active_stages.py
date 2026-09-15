"""active_file_stages: returns only the current file being processed by RUNNING jobs + its stage."""
from src.domain.rag.index_job_manager import IndexingJobManager, IndexJobState, JobStatus


def _state(**kw):
    s = IndexJobState(job_id=kw.get("job_id", "j"), folder_id=kw.get("folder_id", 1))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_only_running_with_current_file():
    mgr = IndexingJobManager()
    mgr._jobs = {
        "j1": _state(job_id="j1", status=JobStatus.RUNNING, current_file_id="f1",
                     current_file_stage="embedding", current_file_chunks_done=2,
                     current_file_chunks_total=8),
        "j2": _state(job_id="j2", status=JobStatus.SUCCEEDED, current_file_id="f2"),
        "j3": _state(job_id="j3", status=JobStatus.RUNNING, current_file_id=None),
    }
    out = mgr.active_file_stages()
    assert set(out) == {"f1"}
    assert out["f1"]["stage"] == "embedding" and out["f1"]["done"] == 2
