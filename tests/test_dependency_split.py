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
    assert {"llama-index-embeddings-huggingface", "transformers", "ckip-transformers"} <= local


def test_no_in_process_pdf_or_image_models():
    """1.5: PDFs and images are read by a vision model or from the text layer. No docling
    model stack (layout, TableFormer, RapidOCR / EasyOCR) in any install, and no torchvision /
    torchaudio (only those models used them)."""
    project = _project()
    specs = list(project["dependencies"]) + [d for ds in project["optional-dependencies"].values() for d in ds]
    specs += [d for ds in tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml")
                                         .read_text(encoding="utf-8")).get("dependency-groups", {}).values()
              for d in ds if isinstance(d, str)]
    names = {_name(d) for d in specs}
    assert not names & {"docling", "docling-ibm-models", "rapidocr", "easyocr", "torchvision", "torchaudio"}
    lock = (Path(__file__).resolve().parents[1] / "uv.lock").read_text(encoding="utf-8")
    for pkg in ("docling-ibm-models", "rapidocr", "easyocr", "torchvision", "torchaudio"):
        assert f'name = "{pkg}"' not in lock, pkg


def test_no_in_process_speech_to_text():
    """1.5: speech-to-text is an OpenAI-compatible endpoint. No Whisper, FireRedASR or kaldi
    in any install, not even through docling's [asr] extra, and nothing vendored."""
    project = _project()
    specs = list(project["dependencies"]) + [d for ds in project["optional-dependencies"].values() for d in ds]
    names = {_name(d) for d in specs}
    assert not names & {"fireredasr", "openai-whisper", "kaldiio", "kaldi-native-fbank", "faster-whisper"}
    assert not any(re.search(r"^docling\[[^]]*asr", d.replace(" ", "")) for d in specs)
    root = Path(__file__).resolve().parents[1]
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    for pkg in ("fireredasr", "openai-whisper", "kaldiio", "mlx-whisper", "whisper-s2t-reborn"):
        assert f'name = "{pkg}"' not in lock, pkg
    assert not (root / "vendor" / "fireredasr_src").exists()


def test_core_renders_pdfs_with_pypdfium2_not_agpl_pymupdf():
    # Folio is MIT: PDF page counts and rendering use PDFium (Apache-2.0 / BSD-3-Clause),
    # declared directly so the cloud profile does not rely on docling's transitive copy.
    # PyMuPDF is AGPL-3.0 and must not come back.
    core = {_name(d) for d in _project()["dependencies"]}
    assert "pypdfium2" in core
    assert not core & {"pymupdf", "fitz", "pymupdfb"}


def test_only_torch_comes_from_another_index():
    """Every package but torch comes from PyPI. torch is the only one mapped to PyTorch's CPU
    index (explicit); until 1.5.0 a non-explicit AMD index locked jinja2, markupsafe, mpmath,
    networkx and sympy to it, and its markupsafe had no arm64 wheel, which broke the
    linux/arm64 image."""
    lock = tomllib.loads((Path(__file__).resolve().parents[1] / "uv.lock").read_text())
    stray = sorted(
        f"{p['name']} {p['version']}" for p in lock["package"]
        if "pypi.org" not in p.get("source", {}).get("registry", "pypi.org")
        and p["name"] != "torch"
    )
    assert stray == [], f"locked to a non-PyPI index: {stray}"
