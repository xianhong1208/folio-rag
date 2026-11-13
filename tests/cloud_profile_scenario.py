"""Subprocess scenario for tests/test_cloud_profile_no_local_models.py — not a test module.

Runs the cloud profile end to end against the in-process fake model server and prints one
JSON line: what worked, which watched modules anybody *tried* to import (and from where),
and which ended up in sys.modules.

    python [-S] tests/cloud_profile_scenario.py <workdir> <full|query>

full  — boot (config + create_app + startup decisions + RAGContext), index-time work for a
        .docx (docling Office backend), a PDF and a PNG (cloud vision), a WAV (cloud STT),
        leaf split, contextual prefixes and embeddings; then the query-time work.
query — boot + query-time work only (query embedding, BM25 segmentation with the cloud
        default bigram segmenter, rerank, answer).

The pgvector write/read itself is not exercised (it needs Postgres); it is SQL through
psycopg2 and imports no model library.
"""

import importlib.abc
import json
import math
import os
import shutil
import struct
import sys
import traceback
import wave

WATCH = {"torch", "torchvision", "torchaudio", "transformers", "sentence_transformers",
         "docling_ibm_models", "ckip_transformers", "whisper", "fireredasr", "rapidocr",
         "kaldiio", "kaldi_native_fbank"}
HF_EMBED = "llama_index.embeddings.huggingface"
attempts = []


class _Recorder(importlib.abc.MetaPathFinder):
    """Records every attempt to import a watched module; never blocks anything."""

    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        hit = root if root in WATCH else (HF_EMBED if name.startswith(HF_EMBED) else None)
        if hit:
            frames = [f for f in traceback.extract_stack()[:-1]
                      if "importlib" not in f.filename and "<frozen" not in f.filename]
            attempts.append({"module": hit, "file": frames[-1].filename, "line": frames[-1].lineno})
        return None


sys.meta_path.insert(0, _Recorder())


def _loaded():
    return sorted(m for m in sys.modules if m.split(".")[0] in WATCH or m.startswith(HF_EMBED))


CONFIG = """\
server: {{host: 127.0.0.1, port: 5999, transport: http}}
database: {{url: "postgresql://u:p@127.0.0.1:1/none", echo: false, pool_size: 1, max_overflow: 0, pool_timeout: 1, pool_recycle: 60}}
auth: {{enabled: false}}
logging: {{level: INFO, format: x, file: null, max_size: 1MB, backup_count: 1}}
app: {{name: CloudProof}}
modules: {{enabled: [rag_query, rag, admin, media]}}
rag:
  profile: cloud
  embedding: {{provider: openai-compatible, model: emb, dimension: 8, base_url: "{url}/v1", api_key: "${{CLOUD_KEY}}"}}
  llm: {{provider: openai-compatible, model: chat, base_url: "{url}/v1", api_key: "${{CLOUD_KEY}}"}}
  rerank: {{enabled: true, api: cohere, model: rr, base_url: "{url}", api_key: "${{CLOUD_KEY}}"}}
  asr: {{provider: openai-compatible, model: whisper-1, base_url: "{url}/v1", api_key: "${{CLOUD_KEY}}"}}
  parsing:
    vision: {{model: vlm, base_url: "{url}/v1", api_key: "${{CLOUD_KEY}}"}}
  contextual_retrieval: {{enabled: true}}
  chunking: {{hierarchy_sizes: [1024, 256]}}
  retrieval: {{default_top_k: 5, default_similarity_cutoff: 0.1, hybrid_search: true}}
  vector_store: {{type: pgvector, table_prefix: vs}}
"""


def main(workdir: str, mode: str) -> dict:
    import asyncio
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    from tests.fake_model_server import FakeModelServer

    os.chdir(workdir)
    os.environ["CLOUD_KEY"] = "sk-test"
    os.environ["HF_HUB_OFFLINE"] = "1"  # no tokenizer download: exercise the estimate path
    steps = {}

    with FakeModelServer(dim=8) as srv:
        def chat(body):
            content = body["messages"][-1]["content"]
            if isinstance(content, list):  # vision request
                return "# Scanned Report\n\nThe scanned page body."
            if "JSON" in content:
                return "[true]"
            return "A short context line." if "<chunk>" in content else "An answer [1]."
        srv.chat_reply = chat
        from tests.config_helpers import write_config_dir
        write_config_dir("config", CONFIG.format(url=srv.url))

        # ---- boot
        from src.config.config_manager import Config
        Config.set_config("config")
        cfg = Config.get_config_model()
        assert cfg is not None and cfg.rag.profile == "cloud", "cloud config failed to load"
        import app
        app.create_app(cfg, transport="http")
        from src.config.roles import docling_warmup_needed, startup_lines
        steps["warmup_skipped"] = not docling_warmup_needed(cfg.rag)
        steps["startup_lines"] = len(startup_lines(cfg.rag))
        from src.adapter.rag_context import RAGContext
        ctx = RAGContext.from_config()
        indexer = ctx.indexer

        # ---- index-time work
        if mode == "full":
            import docx
            import fitz
            from PIL import Image
            d = docx.Document()
            d.add_heading("Travel Policy", 1)
            d.add_paragraph("Trips must be approved by a manager before booking.")
            d.save("policy.docx")
            pdf = fitz.open()
            pdf.new_page().insert_text((72, 72), "scanned")
            pdf.save("scan.pdf")
            Image.new("RGB", (64, 64), "white").save("photo.png")
            files = [("policy.docx", "policy.docx"), ("scan.pdf", "scan.pdf"), ("photo.png", "photo.png")]
            if shutil.which("ffmpeg"):
                with wave.open("talk.wav", "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(b"".join(struct.pack("<h", int(9000 * math.sin(i / 6)))
                                           for i in range(16000)))
                files.append(("talk.wav", "talk.wav"))
            loader = indexer._document_loader
            leaves = []
            with patch("src.domain.rag.document_loader.FileStorage.resolve_path",
                       side_effect=lambda p: Path(p)):
                for i, (path, name) in enumerate(files):
                    doc = loader.load(path, str(i), name)
                    steps[f"loaded:{name}"] = len(doc.text)
                    leaves += indexer._leaf_splitter.split_into_leaf_documents(doc)
            steps["pdf_page_no"] = next(
                (lf.metadata.get("page_no") for lf in leaves if "scanned page body" in lf.text), None)
            texts = [lf.text for lf in leaves]
            prefixes = asyncio.run(indexer._context_generator.generate_prefixes(
                texts, "\n".join(texts), "all"))
            steps["contextualized"] = sum(p == "A short context line." for p in prefixes)
            from src.domain.rag.chunk_identity import compose_retrieval_text
            retrieval = [compose_retrieval_text(p, None, t) for p, t in zip(prefixes, texts)]
            steps["embedded"] = len(indexer._embed_model.get_text_embedding_batch(retrieval))

        # ---- query-time work
        steps["query_dim"] = len(indexer._embed_model.get_query_embedding("travel approval"))
        # Keyword search: the profile's default segmenter (bigram under cloud — no model),
        # the same call indexing, Rebuild FTS and the BM25 query side make.
        from src.domain.rag.text_segmenter import configured_segmenter_name, get_segmenter
        from src.domain.rag.hybrid_search import query_lexemes
        steps["segmenter"] = configured_segmenter_name()
        steps["segmented"] = get_segmenter().segment("差旅需要主管核准")
        steps["query_lexemes"] = query_lexemes(get_segmenter().segment("主管核准"))
        nodes = [SimpleNamespace(text=t, score=0.0) for t in ("first", "second")]
        steps["reranked"] = [n.text for n in asyncio.run(ctx.reranker.arerank("q", nodes))]
        from src.api.router.admin_manage import _generate_answer
        steps["answer"] = _generate_answer("Who approves trips?", [{"text": "Managers approve.", "score": 1}])["answer"]
        steps["requests"] = len(srv.requests)

    return {"steps": steps, "attempts": attempts, "loaded": _loaded()}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = main(sys.argv[1], sys.argv[2])
    print("SCENARIO_RESULT " + json.dumps(result, ensure_ascii=False, default=str))
