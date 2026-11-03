"""Fetch Taiwanese laws from law.moj.gov.tw and save them as Markdown eval documents.

Taiwanese statutes and regulations are not protected by copyright (Copyright Act
Art. 9), so their full texts can live in this repository as retrieval-eval material.

Usage:
    python scripts/eval_fetch_laws.py                 # fetch every law in LAWS
    python scripts/eval_fetch_laws.py --html-dir DIR  # convert already-downloaded LawAll pages

The output (evals/rag/documents/<slug>.md) is committed; re-running only matters when
a law is amended. The amendment date is written into each file.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "evals" / "rag" / "documents"
URL = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={pcode}"

# pcode -> output slug
LAWS = {
    "N0030001": "tw-labor-standards-act",
    "N0030002": "tw-labor-standards-act-enforcement-rules",
    "N0030014": "tw-gender-equality-in-employment-act",
    "I0050021": "tw-personal-data-protection-act",
    "I0050022": "tw-personal-data-protection-act-enforcement-rules",
    "N0030006": "tw-rules-of-leave-taking-for-workers",
}

_TAG = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    return html.unescape(_TAG.sub("", fragment)).strip()


def law_html_to_markdown(page: str, pcode: str) -> str:
    """Convert a LawAll.aspx page into Markdown: chapters as ##, articles as ###."""
    name = _text(re.search(r'id="hlLawName"[^>]*>(.*?)</a>', page, re.S).group(1))
    date_m = re.search(r"(修正日期|公布日期|發布日期)：</th>\s*<td>(.*?)</td>", page, re.S)
    date = f"{date_m.group(1)}{_text(date_m.group(2))}" if date_m else ""
    body = page[page.find('class="law-reg-content"'):] if 'law-reg-content' in page else page
    out = [f"# {name}", "",
           f"Source: {URL.format(pcode=pcode)} (全國法規資料庫, {date}). "
           "Public domain in Taiwan: not protected by copyright (Copyright Act Art. 9).", ""]
    token = re.compile(
        r'<div class="h3 char-\d+">(?P<chapter>.*?)</div>'
        r'|<div class="col-no">\s*<a[^>]*>(?P<article>.*?)</a></div>'
        r'|<div class="line-\d+[^"]*">(?P<line>.*?)</div>', re.S)
    for m in token.finditer(body):
        if m.group("chapter") is not None:
            out += ["", f"## {re.sub(r'\s+', ' ', _text(m.group('chapter')))}", ""]
        elif m.group("article") is not None:
            out += ["", f"### {_text(m.group('article'))}"]
        else:
            line = _text(m.group("line"))
            if line:
                out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-dir", help="read <pcode>.html from here instead of fetching")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pcode, slug in LAWS.items():
        if args.html_dir:
            page = (Path(args.html_dir) / f"{pcode}.html").read_text(encoding="utf-8")
        else:
            req = urllib.request.Request(URL.format(pcode=pcode),
                                         headers={"User-Agent": "folio-eval/1.0"})
            page = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        md = law_html_to_markdown(page, pcode)
        (OUT_DIR / f"{slug}.md").write_text(md, encoding="utf-8")
        print(f"{slug}.md: {len(md)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
