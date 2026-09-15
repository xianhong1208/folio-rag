
"""Automated import-cycle guard: every module under src/ has zero circular dependencies (A→B→A permanently blocked).

Guards two levels, both required to be cycle-free:
1. Top-level imports (an import-time cycle crashes at startup).
2. Including in-function lazy imports (runtime-coupling cycles — lazy imports
   are often used to "work around" an import-time cycle, which merely defers the
   crash; architecturally it is still A↔B).

This test pins the current cycle-free state: adding any back-edge (including one
hidden behind a lazy import) turns CI red immediately and prints the full cycle
path. TYPE_CHECKING blocks are excluded (pure typing, no runtime coupling).
"""

import ast
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"


def _modules() -> dict:
    return {
        ".".join(p.relative_to(_SRC.parent).with_suffix("").parts): p
        for p in _SRC.rglob("*.py")
    }


def _edges_of(path: Path, dotted: str, mods: dict, include_lazy: bool) -> set:
    def resolve(name):
        if name in mods:
            return name
        if f"{name}.__init__" in mods:
            return f"{name}.__init__"
        return None

    out = set()

    class V(ast.NodeVisitor):
        def visit_If(self, node):
            if "TYPE_CHECKING" in ast.unparse(node.test):
                for n in node.orelse:
                    self.visit(n)
                return
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if include_lazy:
                self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for a in node.names:
                r = resolve(a.name)
                if r:
                    out.add(r)

        def visit_ImportFrom(self, node):
            if node.module:
                r = resolve(node.module)
                if r:
                    out.add(r)
                else:
                    for a in node.names:
                        r2 = resolve(f"{node.module}.{a.name}")
                        if r2:
                            out.add(r2)

    V().visit(ast.parse(path.read_text(encoding="utf-8")))
    out.discard(dotted)
    return out


def _find_cycles(graph: dict) -> list:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    cycles = []

    def dfs(u, stack):
        color[u] = GRAY
        stack.append(u)
        for v in graph.get(u, ()):
            if color[v] == GRAY:
                cycles.append(stack[stack.index(v):] + [v])
            elif color[v] == WHITE:
                dfs(v, stack)
        stack.pop()
        color[u] = BLACK

    for m in graph:
        if color[m] == 0:
            dfs(m, [])
    return cycles


def _src_cycles(include_lazy: bool) -> list:
    mods = _modules()
    graph = {m: _edges_of(p, m, mods, include_lazy) for m, p in mods.items()}
    return _find_cycles(graph)


def test_cycle_detector_catches_planted_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}
    cycles = _find_cycles(graph)
    assert cycles, "偵測器抓不到自捏的 a→b→c→a 環"


def test_cycle_detector_clean_dag_passes():
    assert _find_cycles({"a": {"b", "c"}, "b": {"c"}, "c": set()}) == []


def test_no_import_time_cycles():
    """Zero top-level import cycles — a cycle crashes at startup; the hardest invariant."""
    cycles = _src_cycles(include_lazy=False)
    assert not cycles, "import-time 環:\n" + "\n".join(" → ".join(c) for c in cycles)


def test_no_runtime_lazy_import_cycles():
    """Zero cycles including lazy imports — lazy is not a license to work around a cycle, only to hide A↔B inside a function."""
    cycles = _src_cycles(include_lazy=True)
    assert not cycles, "執行期(lazy)環:\n" + "\n".join(" → ".join(c) for c in cycles)


def test_graph_is_nontrivial():
    """Sanity guard: the graph must actually resolve many modules and edges (not scan nothing)."""
    mods = _modules()
    assert len(mods) > 30, f"只解析到 {len(mods)} 個模組,src 掃描可能壞了"
    graph = {m: _edges_of(p, m, mods, True) for m, p in mods.items()}
    n_edges = sum(len(v) for v in graph.values())
    assert n_edges > 50, f"只解析到 {n_edges} 條邊,import 解析可能壞了"
