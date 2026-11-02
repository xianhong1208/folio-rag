"""Proof: under ``rag.profile: cloud`` the server never imports a local model stack.

Two subprocess runs of tests/cloud_profile_scenario.py against the in-process fake
model server (no network, no keys):

1. **Core install, full pipeline.** The interpreter runs with ``-S`` and a site-packages
   view that leaves out everything the ``local`` extra provides (torch, transformers,
   sentence-transformers, docling-ibm-models, CKIP, whisper, FireRedASR, RapidOCR, kaldi,
   llama-index-embeddings-huggingface) — exactly what ``uv sync`` without ``--extra local``
   installs. Boot, a .docx, a PDF and a PNG through the cloud vision model, a WAV through
   cloud STT, leaf splitting, contextual prefixes, embeddings, query embedding, BM25
   segmentation, rerank and an answer must all work; no file of *ours* may even try to
   import those modules (third-party optional imports, e.g. docling's guarded
   ``try: import transformers``, are allowed to probe and fail), and none may be loaded.

2. **Everything installed, boot + query.** With the full local stack importable, booting
   and serving queries under the cloud profile must still load none of it.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCENARIO = REPO / "tests" / "cloud_profile_scenario.py"

# What the `local` extra adds (import names / distribution names).
LOCAL_MODULES = {"torch", "torchvision", "torchaudio", "triton", "torchgen", "functorch",
                 "transformers", "sentence_transformers", "docling_ibm_models",
                 "ckip_transformers", "whisper", "fireredasr", "rapidocr", "kaldiio",
                 "kaldi_native_fbank"}
LOCAL_DISTS = {"torch", "torchvision", "torchaudio", "triton", "transformers",
               "sentence-transformers", "docling-ibm-models", "ckip-transformers",
               "openai-whisper", "fireredasr", "rapidocr", "kaldiio", "kaldi-native-fbank",
               "llama-index-embeddings-huggingface"}
LOCAL_SUBPATHS = {"llama_index/embeddings/huggingface"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _mirror(src: Path, dst: Path, rel: str = "") -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        r = f"{rel}/{entry.name}".lstrip("/")
        if not rel:
            if entry.name.endswith(".dist-info"):
                if _norm(entry.name[:-len(".dist-info")].rsplit("-", 1)[0]) in LOCAL_DISTS:
                    continue
            elif entry.name.split(".")[0] in LOCAL_MODULES:
                continue
        if r in LOCAL_SUBPATHS:
            continue
        if any(p.startswith(r + "/") for p in LOCAL_SUBPATHS):
            _mirror(entry, dst / entry.name, r)
            continue
        (dst / entry.name).symlink_to(entry)


@pytest.fixture(scope="module")
def core_site_packages(tmp_path_factory):
    purelib = Path(sysconfig.get_paths()["purelib"])
    if not (purelib / "docling").is_dir():
        pytest.skip("site-packages layout not recognised")
    out = tmp_path_factory.mktemp("core_site_packages")
    _mirror(purelib, out)
    return out


def _run(args, env_path: str, workdir: Path, extra_flags=()):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    env["PYTHONPATH"] = env_path
    proc = subprocess.run([sys.executable, *extra_flags, str(SCENARIO), str(workdir), *args],
                          capture_output=True, text=True, timeout=600, env=env, cwd=str(workdir))
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("SCENARIO_RESULT ")), None)
    assert line, f"scenario failed (rc={proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-5000:]}"
    return json.loads(line[len("SCENARIO_RESULT "):])


def _ours(attempts):
    """Import attempts made by this repository's code (not tests, not third-party)."""
    root = str(REPO) + os.sep
    return [a for a in attempts
            if a["file"].startswith(root) and os.sep + "tests" + os.sep not in a["file"]]


def test_core_install_runs_the_whole_cloud_pipeline(core_site_packages, tmp_path):
    result = _run(["full"], f"{core_site_packages}{os.pathsep}{REPO}", tmp_path, extra_flags=("-S",))
    steps = result["steps"]

    assert steps["warmup_skipped"] is True
    assert steps["loaded:policy.docx"] > 0 and steps["loaded:scan.pdf"] > 0
    assert steps["loaded:photo.png"] > 0
    if shutil.which("ffmpeg"):
        assert steps["loaded:talk.wav"] > 0
    assert steps["pdf_page_no"] == 1          # page provenance survives the cloud path
    assert steps["contextualized"] > 0 and steps["embedded"] > 0
    assert steps["query_dim"] == 8
    assert steps["segmented"] == "差旅需要主管核准"  # CKIP not loaded: unsegmented fallback
    assert steps["reranked"] == ["first", "second"]
    assert steps["answer"] == "An answer [1]."

    assert _ours(result["attempts"]) == [], "our code tried to import a local model stack"
    assert result["loaded"] == []


def test_everything_installed_cloud_boot_and_query_load_nothing_local(tmp_path):
    result = _run(["query"], str(REPO), tmp_path)
    assert result["steps"]["warmup_skipped"] is True
    assert result["steps"]["answer"] == "An answer [1]."
    assert result["attempts"] == []
    assert result["loaded"] == []
