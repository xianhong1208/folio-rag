"""_prune_stale: staleness handling for a cached eval set.

Pure logic (no DB): _prune_stale's body does not use the `folder` argument. The
module is a script (loaded via importlib in prod), so we load it the same way.

Rule under test: keep survivors for light staleness (report the drop count), but
regenerate — kept is None with a reason — once more than 30% of the gold chunks are
stale, so scores are never computed over a materially smaller, non-comparable set.
"""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "rag_eval.py")


def _load():
    spec = importlib.util.spec_from_file_location("rag_eval_under_test", _PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rag_eval = _load()


def _items(n):
    return [{"gold_node_id": f"g{i}"} for i in range(n)]


def _alive(keep):
    return {f"g{i}" for i in range(keep)}


class TestPruneStale:
    def test_no_stale_keeps_all(self):
        kept, reason, dropped = rag_eval._prune_stale(None, _items(10), _alive(10))
        assert len(kept) == 10 and reason is None and dropped == 0

    def test_light_staleness_keeps_survivors(self):
        # 2/10 = 20% (<= 30%): keep the survivors, report the 2 dropped
        kept, reason, dropped = rag_eval._prune_stale(None, _items(10), _alive(8))
        assert len(kept) == 8 and reason is None and dropped == 2

    def test_boundary_30pct_not_regenerated(self):
        # exactly 3/10 = 30% is not strictly over the threshold -> keep survivors
        kept, reason, dropped = rag_eval._prune_stale(None, _items(10), _alive(7))
        assert len(kept) == 7 and reason is None and dropped == 3

    def test_heavy_staleness_regenerates(self):
        # 4/10 = 40% (> 30%): regenerate (kept None, reason set, dropped reset to 0)
        kept, reason, dropped = rag_eval._prune_stale(None, _items(10), _alive(6))
        assert kept is None and reason and dropped == 0

    def test_all_stale_regenerates(self):
        kept, reason, dropped = rag_eval._prune_stale(None, _items(10), _alive(0))
        assert kept is None and reason and dropped == 0
