"""Guard tests — one CPU install (1.5.1): no GPU dependency groups, torch only in `local`.

Up to 1.5.0 torch picked its GPU build through PEP 735 dependency groups (`cuda`, `cpu`,
`rocm-r714`), each mapped to its own wheel index and kept apart by `[tool.uv] conflicts`.
1.5.1 runs only CKIP and HuggingFace embeddings in-process, both on the CPU, so there is one
torch: the CPU build from PyTorch's CPU index, installed by `uv sync --extra local`. These
tests keep the GPU groups, their indexes and their conflicts from coming back, and keep the
lock free of CUDA and ROCm packages. Pure parsing of pyproject.toml and uv.lock.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _name(spec: str) -> str:
    """'torch[device-all]==2.13.0+rocm7.14.0' → 'torch'."""
    return re.match(r"\s*([A-Za-z0-9._-]+)", spec).group(1).lower()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def test_no_gpu_dependency_groups(pyproject):
    """No dependency group installs torch (or anything else): `--group cuda|rocm-r714|cpu` is
    gone, and an old command fails at once instead of installing a second torch."""
    groups = pyproject.get("dependency-groups", {})
    assert not groups.keys() & {"cuda", "cpu", "rocm-r714", "rocm-r713"}, sorted(groups)
    with_torch = [g for g, specs in groups.items() if any(_name(s) == "torch" for s in specs if isinstance(s, str))]
    assert with_torch == [], f"torch must come from the local extra, not a group: {with_torch}"
    assert "conflicts" not in pyproject["tool"]["uv"], "no mutually exclusive groups are left to guard"


def test_torch_only_in_the_local_extra(pyproject):
    project = pyproject["project"]
    assert "torch" not in {_name(d) for d in project["dependencies"]}
    for extra, specs in project["optional-dependencies"].items():
        assert ("torch" in {_name(d) for d in specs}) == (extra == "local"), extra


def test_torch_comes_from_the_cpu_index_only(pyproject):
    """One explicit index (PyTorch's CPU wheels) and torch the only package mapped to it. No
    AMD or CUDA index; everything else, torch's own dependencies included, comes from PyPI."""
    uv = pyproject["tool"]["uv"]
    indexes = uv.get("index", [])
    assert [(i["url"].rstrip("/"), i.get("explicit")) for i in indexes] == [(CPU_INDEX, True)], indexes
    assert uv.get("sources") == {"torch": {"index": indexes[0]["name"]}}
    assert "index-strategy" not in uv, "unsafe-best-match was only needed for the AMD index"


def test_lock_has_no_gpu_packages(lock):
    names = {p["name"] for p in lock["package"]}
    gpu = sorted(n for n in names if re.match(r"^(nvidia-.+|cuda-.+|triton|pytorch-triton.*|rocm(-.+)?|amd-torch-.+)$", n))
    assert gpu == [], f"CUDA / ROCm packages in the lock: {gpu}"


def test_locked_torch_is_the_cpu_build(lock):
    """Linux x86_64 and aarch64 get +cpu wheels; macOS gets the CPU index's arm64 wheel (Apple
    builds carry no CUDA). Both from the CPU index."""
    torches = [p for p in lock["package"] if p["name"] == "torch"]
    assert torches
    files = []
    for p in torches:
        assert p["source"] == {"registry": CPU_INDEX}, p["source"]
        assert p["version"].startswith("2.14."), p["version"]
        assert "+" not in p["version"] or p["version"].endswith("+cpu"), p["version"]
        files += [w["url"].rsplit("/", 1)[1].replace("%2B", "+") for w in p.get("wheels", [])]
    for platform in ("manylinux_2_28_x86_64", "manylinux_2_28_aarch64", "macosx_14_0_arm64"):
        assert any(f.endswith(f"cp312-cp312-{platform}.whl") for f in files), (platform, files)
    assert all("+cpu" in f for f in files if "linux" in f), files
