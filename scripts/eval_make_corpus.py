"""Build the generated part of the retrieval-eval corpus (evals/rag/documents/).

Two kinds of files are generated here; everything else in the corpus is either
fetched (scripts/eval_fetch_laws.py, U.S. government pages) or hand-written:

  * PDFs with tables: evals/rag/sources/*.md -> pandoc -> .docx -> LibreOffice -> .pdf
  * synthetic zh-TW meeting transcripts, written by the LLM from the briefs below.
    Responses are cached in evals/rag/llm_cache.jsonl, so a re-run reproduces the
    same files without calling the LLM.

Usage:
    python scripts/eval_make_corpus.py [--pdf] [--transcripts] [--offline]
"""
from __future__ import annotations

import os
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evals.llm_cache import CachedLLM  # noqa: E402

DOCS = REPO / "evals" / "rag" / "documents"
SOURCES = REPO / "evals" / "rag" / "sources"
CACHE = REPO / "evals" / "rag" / "llm_cache.jsonl"

LLM_BASE_URL = "http://localhost:5052/v1"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_API_KEY = os.environ.get("EVAL_LLM_API_KEY", "not-needed")  # set EVAL_LLM_API_KEY for a keyed endpoint

SYNTHETIC_NOTE = ("> 合成資料（SYNTHETIC）：本逐字稿由 LLM（{model}）依 Folio 評測集的設定產生，"
                  "人物、公司（港光科技，Harborlight Technologies）與事件皆為虛構。授權：MIT。\n")

TRANSCRIPTS = {
    "zh-meeting-2026-02-aster-k2-launch.md": """\
會議名稱：Aster K2 智慧快煮壺上市籌備會議（第 3 次）
日期：2026 年 2 月 24 日（星期二）14:00–15:30，地點：台北總部 8F 大會議室
出席：產品經理 林雅婷（主持）、行銷部經理 陳志豪、業務部協理 王美玲、研發部韌體組長 張家銘、客服部主任 黃淑芬、紀錄 李佳穎
議程與必須出現的具體內容：
1. 上市日期：原定 3 月 10 日，因 NCC 審驗文件補件延後，決議改為 3 月 17 日上市。
2. 定價：建議售價 3,980 元；上市首月早鳥價 3,580 元，限量 2,000 台。
3. 通路：首波通路為官網、momo 與 PChome，實體通路（燦坤）延至 5 月。
4. 韌體：張家銘報告 1.0.8 版修正 Wi-Fi 斷線錯誤碼 E4 誤報，預計 3 月 5 日推送；5 GHz 網路仍不支援。
5. 客服：黃淑芬要求上市前完成 FAQ 30 題，並增設週六 9:00–13:00 客服時段。
6. 行銷：陳志豪報告預算 380 萬元，其中 KOL 合作 120 萬元，目標上市第一季售出 3,000 台。
7. 臨時動議：王美玲提議贈送除垢用檸檬酸包，決議由業務部評估成本後下次會議報告。
待辦事項請列出負責人與期限。""",
    "zh-meeting-2026-05-hr-policy-review.md": """\
會議名稱：人資制度檢討會議——特別休假、加班與混合辦公
日期：2026 年 5 月 12 日（星期二）10:00–11:45，地點：線上會議（Teams）
出席：人資部經理 周怡君（主持）、法務 吳承翰、研發部經理 鄭博文、財務部副理 許雅雯、員工代表 劉冠宇、紀錄 蔡欣怡
議程與必須出現的具體內容：
1. 特休未休工資：法務吳承翰說明依勞動基準法第 38 條，年度終結或契約終止而未休之日數，雇主應發給工資；公司決議 2026 年起特休可遞延至次一年度，遞延未休者於次年度終結時結清。
2. 加班：研發部 4 月平均每人加班 31 小時，鄭博文說明為 AirPure 5 開發時程；決議每月加班超過 40 小時須經副總核准，並重申延長工作時間連同正常工時一日不得超過 12 小時。
3. 混合辦公：決議自 4 月 1 日起實施的混合辦公政策（每週 3 天進辦公室、2 天遠端，週二週四為全員到班日）維持不變；網路補助每月 800 元。劉冠宇反映希望改為週一週五遠端，不同意見列入下季評估。
4. 家庭照顧假：周怡君說明依性別平等工作法，家庭照顧假全年以七日為限，併入事假計算；公司將開放以小時為單位請假。
5. 預算：許雅雯報告人資部 2026 年預算 705 萬元，混合辦公居家設備補助每人一次 6,000 元已發放 182 人。
待辦事項請列出負責人與期限。""",
    "zh-meeting-2026-08-security-incident-review.md": """\
會議名稱：資安事件事後檢討會議（事件編號 INC-2026-031）
日期：2026 年 8 月 20 日（星期四）16:00–17:30，地點：台北總部 6F 資安戰情室
出席：資安長 蕭志偉（主持，事件指揮官）、雲端平台組長 江柏霖、法務 吳承翰、客服部主任 黃淑芬、公關 葉品妤、紀錄 洪子晴
議程與必須出現的具體內容：
1. 事件經過：8 月 11 日凌晨 2:40 監控發現 Harborlight Home 後台管理帳號異常登入，來源為境外 IP；原因是一名工程師在外部論壇與公司帳號重複使用同一組密碼，且該帳號未啟用多因素驗證（MFA）。
2. 嚴重等級：初判 SEV-2，8 月 12 日確認約 1,240 筆會員的姓名、電子郵件與裝置序號遭查詢匯出，升級為 SEV-1。
3. 處置：江柏霖報告已重設所有管理帳號密碼、撤銷 API 金鑰並強制所有後台帳號啟用 MFA，相關日誌保存三年。
4. 通知：法務吳承翰說明依個人資料保護法第 12 條，查明後應以適當方式通知當事人；公司已於 8 月 14 日寄送通知信，並向主管機關通報。
5. 客服：黃淑芬報告事件後三天內收到 312 件詢問，已建立專屬 FAQ。
6. 改善措施：導入密碼管理器（全公司）、每季資安演練、9 月底前完成全員密碼教育訓練；事件檢討報告於 10 個工作天內完成。
待辦事項請列出負責人與期限。""",
}

TRANSCRIPT_PROMPT = """請依照下列會議設定，撰寫一份完整、自然的繁體中文（台灣用語）會議逐字稿。

要求：
- 以 Markdown 撰寫：第一行為「# <會議名稱>」，接著列出日期、地點、出席者，再以「## 議程一：…」等標題分段。
- 每段以「發言人姓名：發言內容」的形式呈現多輪對話，口語自然，可有附和、追問與小插曲，但所有設定中的數字、日期、決議都必須正確出現。
- 不要加入設定以外的重大決議或其他數字，也不要加入與會議無關的旁白或括號註記。
- 全文只能使用繁體中文字，絕對不可出現簡體字（例如寫「張家銘」「網路」「韌體」，不可寫「张家铭」「网络」「固件」）。
- 最後加上「## 決議與待辦事項」段落，以條列列出負責人與期限。
- 長度約 2,500 到 3,500 個中文字。只輸出逐字稿本身。

會議設定：
{brief}
"""


def build_pdfs() -> None:
    soffice = shutil.which("soffice")
    if not soffice or not shutil.which("pandoc"):
        raise SystemExit("pandoc and LibreOffice (soffice) are required to build the PDFs")
    with tempfile.TemporaryDirectory() as tmp:
        for md in sorted(SOURCES.glob("*.md")):
            docx = Path(tmp) / f"{md.stem}.docx"
            subprocess.run(["pandoc", str(md), "-o", str(docx)], check=True)
            subprocess.run([soffice, f"-env:UserInstallation=file://{tmp}/lo", "--headless",
                            "--convert-to", "pdf", "--outdir", str(DOCS), str(docx)],
                           check=True, capture_output=True)
            print(f"{md.stem}.pdf")


# Simplified characters the LLM still slipped into otherwise Traditional text. Fixed
# deterministically so a cached re-run reproduces the committed files exactly.
_TO_TRADITIONAL = str.maketrans({"来": "來", "价": "價", "对": "對", "这": "這", "说": "說",
                                 "们": "們", "会": "會", "为": "為", "时": "時", "发": "發",
                                 "应": "應", "数": "數", "泄": "洩", "铭": "銘"})


def build_transcripts(llm: CachedLLM) -> None:
    for name, brief in TRANSCRIPTS.items():
        text = llm.complete(TRANSCRIPT_PROMPT.format(brief=brief), tag=f"transcript:{name}",
                            max_tokens=8000, temperature=0.7).translate(_TO_TRADITIONAL)
        lines = text.strip().splitlines()
        # Keep the LLM's title line first, then the synthetic-data notice.
        head, rest = (lines[0], lines[1:]) if lines and lines[0].startswith("# ") else ("", lines)
        out = "\n".join(([head, ""] if head else []) + [SYNTHETIC_NOTE.format(model=LLM_MODEL)] + rest)
        (DOCS / name).write_text(out.strip() + "\n", encoding="utf-8")
        print(f"{name}: {len(out)} chars")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="rebuild the PDFs")
    ap.add_argument("--transcripts", action="store_true", help="(re)write the transcripts")
    ap.add_argument("--offline", action="store_true", help="fail instead of calling the LLM")
    args = ap.parse_args()
    if not (args.pdf or args.transcripts):
        args.pdf = args.transcripts = True
    if args.pdf:
        build_pdfs()
    if args.transcripts:
        build_transcripts(CachedLLM(CACHE, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY,
                                    offline=args.offline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
