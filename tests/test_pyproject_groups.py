
"""Guard tests — consistency of the three PyTorch dependency-group lists (PEP 735 + uv).

torch picks its GPU build via a dependency group (cuda / cpu / rocm-r713 /
rocm-r714). For a group to work correctly it must appear in all three places at
once:
  (1) [dependency-groups]   — declares the torch trio's versions
  (2) [tool.uv] conflicts   — mutual-exclusion guard (omit it → two torch sets
                              overwrite each other, and uv does not warn)
  (3) [tool.uv.sources]     — maps torch/torchvision/torchaudio to the right index

The pyproject header states that the conflicts list is the sole guard and uv
will not warn if a group is missing from it. This test automates that sole
guard, so adding a new ROCm build cannot silently miss one place. Pure parsing
of pyproject.toml, no external dependencies.
"""

import re
import tomllib
from pathlib import Path

import pytest

_TORCH_STACK = ("torch", "torchvision", "torchaudio")


def _load_pyproject() -> dict:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def _dep_name(spec: str) -> str:
    """Extract the package name from a dep string: 'torch[device-all]==2.11.0+rocm7.13.0' → 'torch'."""
    m = re.match(r"^\s*([A-Za-z0-9._-]+)", spec)
    return m.group(1).lower() if m else ""


def _torch_groups(data: dict) -> set[str]:
    """Return the groups that declare torch — only these need mutual exclusion + source mapping.

    Non-torch groups (should a dev/lint one be added later) are not pinned by
    this test.
    """
    groups = data.get("dependency-groups", {})
    return {
        name
        for name, specs in groups.items()
        if any(_dep_name(s) == "torch" for s in specs if isinstance(s, str))
    }


def _conflict_groups(data: dict) -> set[str]:
    result: set[str] = set()
    for cluster in data["tool"]["uv"].get("conflicts", []):
        for item in cluster:
            if "group" in item:
                result.add(item["group"])
    return result


def _source_groups(data: dict, package: str) -> set[str]:
    entries = data["tool"]["uv"].get("sources", {}).get(package, [])
    return {e["group"] for e in entries if "group" in e}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return _load_pyproject()


def test_torch_groups_present(pyproject):
    """Precondition: torch groups are actually parsed (otherwise later assertions run empty and pass falsely)."""
    groups = _torch_groups(pyproject)
    assert groups, "沒有解析到任何宣告 torch 的 dependency-group"
    # Should currently cover at least these; adding others won't fail the test, but a missing existing one matters
    assert {"cuda", "cpu"} <= groups, f"預期 cuda/cpu 為 torch group,實得 {groups}"


def test_every_torch_group_in_conflicts(pyproject):
    """Every torch group must be in the conflicts list — omit one and two torch sets install together and overwrite each other."""
    torch_groups = _torch_groups(pyproject)
    conflict_groups = _conflict_groups(pyproject)
    missing = torch_groups - conflict_groups
    assert not missing, (
        f"torch group {sorted(missing)} 不在 [tool.uv] conflicts 清單裡;"
        f"同時啟用會互相覆蓋且 uv 不會警告。請加進互斥清單。"
    )


def test_conflicts_only_reference_real_groups(pyproject):
    """Reverse: conflicts must not reference a nonexistent group (e.g. a leftover old per-gfx alias)."""
    declared = set(pyproject.get("dependency-groups", {}))
    stale = _conflict_groups(pyproject) - declared
    assert not stale, f"conflicts 引用了不存在的 group {sorted(stale)}(可能是刪 group 時漏清)"


@pytest.mark.parametrize("package", _TORCH_STACK)
def test_every_torch_group_has_source_mapping(pyproject, package):
    """Every torch group must have an index source mapping for each of the torch/torchvision/torchaudio trio.

    A missing source mapping → the package resolves from the default PyPI, where
    ROCm's +rocmX.Y.Z local version is not found → sync fails outright (at least
    it breaks on the spot rather than silently using the wrong index).
    """
    torch_groups = _torch_groups(pyproject)
    mapped = _source_groups(pyproject, package)
    missing = torch_groups - mapped
    assert not missing, (
        f"{package} 缺少 group {sorted(missing)} 的 [tool.uv.sources] index 對應"
    )
