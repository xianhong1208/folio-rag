"""How well a vision model reads PDF pages: character error rate and character F1 against the
public eval PDFs' own text layer (evals/rag/documents/*.pdf, MIT).

Every page is rendered with PDFium (as Folio renders it for rag.parsing.vision) and sent to an
OpenAI-compatible vision model with Folio's own transcription prompt; the reference is the same
page's text layer. ``--degrade`` first turns each render into a cheap office scan (grey, a
1.5 degree skew, 120 dpi, blur, speckles, JPEG quality 35), which is where OCR engines fail.

The model is read from the environment, like Folio's config: VISION_BASE_URL, VISION_MODEL,
VISION_API_KEY (sent as a header, never printed) and VISION_CA_FILE (a private CA such as
Forgeyard's). Example:

    VISION_BASE_URL=https://forgeyard:7860/v1 VISION_MODEL=qwen3-vl-8b \\
    VISION_API_KEY=... VISION_CA_FILE=forgeyard-ca.pem \\
    .venv/bin/python scripts/eval_ocr.py --degrade --out evals/baselines/ocr-<date>.json

Scores are page-level; Markdown table syntax and whitespace are ignored (evals/metrics.py).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.metrics import cer, char_f1, page_text  # noqa: E402

DOCS = ROOT / "evals" / "rag" / "documents"


def pages(dpi: int):
    """(name, png bytes, reference text) for every page of every public eval PDF."""
    import pypdfium2 as pdfium

    from src.domain.rag import pdf_pages

    for pdf in sorted(DOCS.glob("*.pdf")):
        doc = pdfium.PdfDocument(str(pdf))
        try:
            for i in range(len(doc)):
                textpage = doc[i].get_textpage()
                ref = textpage.get_text_range()
                textpage.close()
                yield f"{pdf.stem}-{i + 1}", pdf_pages.render_png(str(pdf), i + 1, dpi), ref
        finally:
            doc.close()


def degrade(png: bytes, seed: int) -> bytes:
    """A cheap office scan of a clean render (deterministic for a seed)."""
    from PIL import Image, ImageFilter

    rnd = random.Random(seed)
    im = Image.open(io.BytesIO(png)).convert("L")
    im = im.rotate(1.5, resample=Image.BICUBIC, expand=True, fillcolor=255)
    w, h = im.size
    im = im.resize((int(w * 0.6), int(h * 0.6)), Image.BILINEAR)
    im = im.filter(ImageFilter.GaussianBlur(0.7))
    px = im.load()
    for _ in range(im.size[0] * im.size[1] // 60):
        px[rnd.randrange(im.size[0]), rnd.randrange(im.size[1])] = rnd.choice((0, 90, 255))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=35)
    out = io.BytesIO()
    Image.open(io.BytesIO(buf.getvalue())).convert("RGB").save(out, "PNG")
    return out.getvalue()


def transcribe(client, model: str, png: bytes, max_tokens: int) -> str:
    from src.domain.rag.cloud_parser import DEFAULT_PROMPT

    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": DEFAULT_PROMPT},
                  {"role": "user", "content": [{"type": "text", "text": "Page 1 of 1."},
                                               {"type": "image_url", "image_url": {"url": uri}}]}])
    return resp.choices[0].message.content or ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--degrade", action="store_true", help="score degraded scans instead of clean renders")
    ap.add_argument("--dpi", type=int, default=150, help="render resolution (Folio's rag.parsing.dpi)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", type=Path, help="write per-page results as JSON")
    ap.add_argument("--save-text", type=Path, help="directory for each page's transcription")
    args = ap.parse_args(argv)

    base_url, model = os.environ.get("VISION_BASE_URL"), os.environ.get("VISION_MODEL")
    if not (base_url and model):
        ap.error("set VISION_BASE_URL and VISION_MODEL (and VISION_API_KEY / VISION_CA_FILE as needed)")
    from src.config.model import VisionModelConfig
    from src.domain.rag.llm_client import create_openai_client

    cfg = VisionModelConfig(model=model, base_url=base_url, api_key=os.environ.get("VISION_API_KEY") or None,
                            ca_file=os.environ.get("VISION_CA_FILE") or None, timeout=300, max_retries=2)
    client = create_openai_client(cfg)
    results = {}
    for n, (name, png, ref) in enumerate(pages(args.dpi)):
        image = degrade(png, seed=7 + n) if args.degrade else png
        t0 = time.time()
        hyp = transcribe(client, model, image, args.max_tokens)
        seconds = time.time() - t0
        if args.save_text:
            args.save_text.mkdir(parents=True, exist_ok=True)
            (args.save_text / f"{name}.md").write_text(hyp, encoding="utf-8")
        results[name] = {"cer": round(cer(page_text(ref), page_text(hyp), normalize=False), 4),
                         "char_f1": round(char_f1(ref, hyp), 4),
                         "ref_chars": len(page_text(ref)), "seconds": round(seconds, 1)}
        print(f"{name:45} CER {results[name]['cer']:.4f}  F1 {results[name]['char_f1']:.4f}  "
              f"{seconds:5.1f}s", flush=True)
    client.close()
    if results:
        mean = {k: round(sum(r[k] for r in results.values()) / len(results), 4) for k in ("cer", "char_f1")}
        print(f"{'mean':45} CER {mean['cer']:.4f}  F1 {mean['char_f1']:.4f}  ({len(results)} pages, "
              f"{'degraded' if args.degrade else 'clean'})")
        if args.out:
            args.out.write_text(json.dumps({"model": model, "degraded": args.degrade, "dpi": args.dpi,
                                            "mean": mean, "pages": results}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
