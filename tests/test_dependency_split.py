"""Guard: the core install stays torch-free; the in-process model stack lives in the
`local` extra (`uv sync` vs `uv sync --extra local`). Pure pyproject parsing.

The runtime side of the same promise is tests/test_cloud_profile_no_local_models.py.
"""

import re
import tomllib
from pathlib import Path

# Packages that pull torch (directly or through transformers / sentence-transformers).
LOCAL_ONLY = {"docling", "llama-index-embeddings-huggingface", "fireredasr", "transformers",
              "ckip-transformers", "sentence-transformers", "torch", "openai-whisper",
              "kaldiio", "kaldi-native-fbank", "sentencepiece"}


def _project():
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def _name(spec: str) -> str:
    return re.match(r"\s*([A-Za-z0-9._-]+)", spec).group(1).lower()


def test_core_dependencies_are_torch_free():
    core = {_name(d) for d in _project()["dependencies"]}
    assert not core & LOCAL_ONLY, f"move to the local extra: {sorted(core & LOCAL_ONLY)}"


def test_core_carries_opencc_for_prefix_script_repair():
    # opencc is a self-contained C++ extension (no torch). The contextual-prefix Simplified ->
    # Traditional repair and script detection (han_script) run in every profile, including
    # cloud, where the prefixes come from a cloud LLM.
    core = {_name(d) for d in _project()["dependencies"]}
    assert "opencc" in core


def test_core_parses_documents_without_models():
    deps = " ".join(_project()["dependencies"])
    assert "docling-slim[" in deps and "tokenizers" in deps and "tree-sitter" in deps
    # docling-slim's `standard` / `models-*` extras are the ones that pull torch
    assert "standard" not in deps and "models-local" not in deps


def test_local_extra_carries_the_model_stack():
    local = {_name(d) for d in _project()["optional-dependencies"]["local"]}
    assert {"docling", "llama-index-embeddings-huggingface", "fireredasr",
            "transformers", "ckip-transformers"} <= local
