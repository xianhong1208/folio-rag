# 升級到 1.5:OCR、版面與語音模型移出 Folio

[English](migrate-1.5.md)

Folio 1.5 不再自己執行任何 OCR、版面、表格或語音轉文字模型。改變有兩項:

- **PDF 與圖片**:設定了 **OpenAI 相容的視覺模型**(`rag.parsing.vision`,例如 Forgeyard 上的 Qwen3-VL)時由它讀取;
  沒有設定時,PDF 從**文字層**讀取,圖片則拒絕上傳。docling 在程序內執行的版面、TableFormer 與 RapidOCR 模型已移除,
  連同 `rag.docling`、`rag.parsing.images`、`rag.parsing.pdf` 的 `local` / `cloud` 值,以及
  `POST /v1/ocr/general` 與 `GET /v1/models`。
- **音訊**:由 OpenAI 相容端點轉錄(見下方[語音轉文字](#語音轉文字))。

docling 本身仍用於 Office、OpenDocument、HTML、Markdown、電子郵件與電子書檔案:這些格式不需要模型。

## PDF 與圖片

原因:在公開評測 PDF 的 8 頁上(乾淨與劣化的掃描、中文與英文),Forgeyard 提供的 Qwen3-VL-8B 每一頁都沒有認錯字,
表格也保留成 Markdown 表格;docling 的 RapidOCR 在劣化的中文頁面上字元 F1 只有 0.935–0.970。視覺模型也讀得懂
OCR 只會拆成零散文字的表格、圖表與手寫,並把 GPU 記憶體、模型檔與 torch 的影像套件移出 Folio。

### 現在檔案怎麼讀

| | 有設定視覺模型 | 沒有視覺模型(預設) |
|---|---|---|
| PDF | 每一頁轉成圖片交給模型轉成 Markdown(表格保持表格),各頁接成一份文件 | 從 PDF 的**文字層**讀取。一般段落會保留;表格變成逐列的純文字 |
| 沒有文字層的 PDF 頁(掃描頁) | 和其他頁一樣轉錄 | 以看得見的標記索引:`[Page 3: no text layer (a scanned or image-only page); set rag.parsing.vision … to read it.]`;檔案會顯示警告並列出頁碼 |
| 每一頁都沒有文字的 PDF | 轉錄 | **索引失敗**,訊息會說要設定 `rag.parsing.vision`。絕不會索引成空白 |
| 圖片(PNG、JPEG、TIFF、WebP、BMP、GIF) | 轉錄 | **拒絕上傳**(422 並說明原因;REST、批次上傳的 `failed_uploads`、主控台都一樣)。之前已存的圖片重新索引時會以同樣的訊息失敗 |
| Office、Markdown、HTML 等 | 和以前一樣由 docling 處理 | 和以前一樣由 docling 處理 |

`rag.parsing.pdf` 決定 PDF 的讀法:

| 值 | PDF 由誰讀 |
|---|---|
| `auto`(預設) | 有設定 `rag.parsing.vision` 時用視覺模型,否則用文字層 |
| `vision` | 一律用視覺模型;沒有視覺模型時 Folio 拒絕啟動 |
| `text` | 一律用文字層,即使有視覺模型(圖片仍交給模型) |

視覺模型有 `model` 與 `base_url` 才算設定完成(provider `openai` 不需要 `base_url`)。三種 profile 行為相同:
`local`、`hybrid`、`cloud` 在設定視覺模型之前,都從文字層讀 PDF。

### 升級後會發生什麼事

| 你的設定 | Folio 的行為 |
|---|---|
| `config/models.yaml` 有 `rag.docling` 區段(`device`、`ocr_enabled`、`ocr_model_scale`、`hf_offline`、`compile_models`) | **拒絕啟動**,訊息指向本指南。請刪除整個區段。若原本依賴 `hf_offline: true`,改在 `.env` 設 `HF_HUB_OFFLINE=1`。 |
| `rag.parsing.pdf: local` 或 `cloud`,或任何 `rag.parsing.images` | **拒絕啟動**,訊息指向本指南。刪掉這幾行(`pdf` 預設為 `auto`)。 |
| `rag.parsing.pdf: vision` 但沒有視覺模型 | 拒絕啟動(主控台也會拒絕這項變更) |
| 主控台設定頁存過同樣的值 | 啟動時刪除並發出警告;之後 `rag.parsing` 依 `config/models.yaml`。 |
| `.env` 裡的 `DOCLING_DEVICE` | 不再使用,請刪除。 |
| 用 `scripts/split_config.py` 轉換 1.4 的 `config.yaml` | 拿掉 `rag.docling`、`rag.parsing.images` 與 `pdf: local`/`cloud`,並印出說明。 |

升級前已索引的 PDF 與圖片保留原本的 chunk。要用新方式讀取,請重新索引資料夾。

### 讓 Folio 使用 Forgeyard 上的視覺模型

1. **部署模型。** 在 Forgeyard 的 Model library 部署 **Qwen3-VL-8B-Instruct**(vLLM),為部署取名,例如 `qwen3-vl-8b`:Folio 要求的模型名稱就是它。顯示卡較小時可用 Qwen3-VL-4B 或 -2B
   (密集表格的準確度較低)。
2. **為 Folio 建立 API 金鑰**:在 Forgeyard 的 *API keys* 頁面建立,只允許使用這一個部署。金鑰只顯示一次,請先複製。
3. **取得 Forgeyard 的 CA**:從 *Trust* 頁面(或 `https://<forgeyard-host>:7860/api/tls/ca.pem`)下載,核對指紋,
   放在 Folio 讀得到的地方,例如 `/etc/folio/forgeyard-ca.pem`。
4. **設定 `.env`**(`config/models.yaml` 已經會讀這些變數):

   ```bash
   VISION_BASE_URL=https://<forgeyard-host>:7860/v1
   VISION_API_KEY=<第 2 步的金鑰>
   VISION_MODEL=qwen3-vl-8b
   VISION_CA_FILE=/etc/folio/forgeyard-ca.pem
   ```

5. **重新啟動 Folio。** 啟動記錄會寫
   `Document parsing (vision) endpoint https://… — openai-compatible / qwen3-vl-8b [pdf=vision, images=vision]`。
6. **測試。** 在 *Settings → Indexing* 的 Document Parsing 按 **Test**(用已存的設定送出一張很小的圖片)。再上傳一份
   掃描的 PDF,搜尋其中一句話。

`VISION_CA_FILE` 和 `ASR_CA_FILE` 一樣是**加進**原本的信任清單。金鑰錯誤時檔案會以
「check rag.parsing.vision base_url / model / api_key」失敗;憑證不被信任時會立刻失敗並指出
`rag.parsing.vision.ca_file`;端點連不上時會寫出端點位址。這些情況都不會留下空白的索引。

**Docker:** 以唯讀方式掛載 CA(`- ./forgeyard-ca.pem:/etc/folio/forgeyard-ca.pem:ro`),並在 `.env` 設定同樣的四個變數。

**速度與成本。** 整個伺服器同時最多處理 `rag.parsing.max_concurrency`(預設 4)頁,每頁附上前一頁的結尾,讓跨頁的表格與
段落保持完整。`max_pages_per_file`(500)與 `max_pages_per_day`(不限)控制成本;`dpi`(200)決定轉成圖片的解析度。
Forgeyard 的金鑰可能限制每個請求的圖片數與影像 token 數:Folio 每個請求只送一頁,長邊最多 2048 像素。

### 設定

| 設定 | 變數 | 預設 | |
|---|---|---|---|
| `rag.parsing.pdf` | — | `auto` | `auto`、`vision` 或 `text`(見上文)。 |
| `rag.parsing.vision.provider` | — | `openai-compatible` | 也可用 `openai`、`vllm`、`ollama`、`azure`。 |
| `rag.parsing.vision.model` | `VISION_MODEL` | 空 | 服務的模型名稱。 |
| `rag.parsing.vision.base_url` | `VISION_BASE_URL` | 空 | 通常以 `/v1` 結尾。 |
| `rag.parsing.vision.api_key` | `VISION_API_KEY` | 空 | 以 `Authorization: Bearer` 送出。 |
| `rag.parsing.vision.ca_file` | `VISION_CA_FILE` | 空 | 端點額外信任的 CA(PEM)。 |
| `rag.parsing.max_concurrency` / `dpi` / `context_chars` | — | `4` / `200` / `800` | 同時處理的頁數、解析度、前一頁的上下文長度。 |
| `rag.parsing.max_pages_per_file` / `max_pages_per_day` | — | `500` / `0`(不限) | 成本上限。 |
| `rag.parsing.on_page_failure` / `min_success_ratio` | — | `mark` / `0.5` | 重試後仍失敗的頁面:加上標記,或讓整個檔案失敗。 |

除了 `ca_file`、`dpi`、`context_chars` 與 `min_success_ratio`,其他設定也能在主控台設定頁修改(需要 `rag:admin`)。

任何以 `image_url` 回應 OpenAI `POST /v1/chat/completions` 的伺服器都可以:vLLM(`Qwen/Qwen3-VL-8B-Instruct`)、
Ollama(`qwen2.5vl`)、OpenAI(`gpt-4o-mini`,provider `openai`)或 Azure OpenAI(provider `azure`)。

### 呼叫 `POST /v1/ocr/general` 與 `GET /v1/models` 的程式

兩者都回 **410 Gone** 並附上本指南的連結。請直接把頁面圖片送給視覺模型:

```bash
IMG=$(base64 -w0 page.png)
curl --cacert forgeyard-ca.pem https://<forgeyard-host>:7860/v1/chat/completions \
  -H "Authorization: Bearer $VISION_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl-8b", "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe all text on this page as Markdown."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}]}'
```

## 語音轉文字

Folio 1.5 不再自己執行語音轉文字模型。在程序內執行的 provider `fireredasr` 與 `docling-whisper` 已移除,
連同它們的相依套件(FireRedASR、Whisper、kaldi)與 `POST /v1/transcriptions/general` 端點。音訊改由
**OpenAI 相容端點** `POST {base_url}/audio/transcriptions` 轉錄,Forgeyard、speaches、vLLM、
faster-whisper-server 與 OpenAI 都提供這個端點。

原因:在 Google FLEURS(中文與英文各 100 段)上,Forgeyard 提供的 whisper-large-v3-turbo 勝過內建的
FireRedASR-AED-L:英文詞錯誤率 5.8% 對 11.1%,中文字錯誤率 9.1% 對 12.0%(純中文片段兩者相同)。讓
whisper 自行偵測語言,和給它語言提示一樣好。改用端點也把 GPU 記憶體、模型檔與 torch 的音訊套件移出 Folio。

### 升級後會發生什麼事

| 你的設定 | Folio 的行為 |
|---|---|
| `config/models.yaml` 裡寫著 `rag.asr.provider: fireredasr` 或 `docling-whisper` | **拒絕啟動**,訊息會指向本指南。請照下文修改設定檔。 |
| 主控台設定頁存過同樣的值 | 啟動時刪除該設定並發出警告;之後 `rag.asr` 依 `config/models.yaml`。 |
| `rag.asr` 沒有 `base_url`(預設) | 照常啟動。**上傳音訊會被拒絕**(422 並說明原因);健康狀態頁的 Speech-to-Text 顯示 *Not set*。其他檔案不受影響。 |
| `rag.asr.enabled: false` | 照常啟動;上傳音訊會被拒絕(422)。 |
| 用 `scripts/split_config.py` 轉換 1.4 的 `config.yaml` | 把舊 provider 改成 `openai-compatible` 並印出說明;接著設定下方的變數。 |

升級前已索引的音訊保留原本的 chunk。若在沒有端點時重新索引已存的音訊檔,會以同樣的訊息失敗,不會被索引成空白。

### 讓 Folio 使用 Forgeyard

1. **部署模型。** 在 Forgeyard 的 Model library 部署 *Whisper large-v3 turbo (faster-whisper)*
   (以 speaches 引擎執行,約需 3 GiB GPU 記憶體,也可以跑在 CPU)。為部署取個名字,例如
   `whisper-large-v3-turbo`:Folio 要求的模型名稱就是它。
2. **為 Folio 建立 API 金鑰。** 在 Forgeyard 的 *API keys* 頁面建立,並只允許它使用這一個部署,這把金鑰就碰不到你的
   其他模型。金鑰只會顯示一次,請先複製。
3. **取得 Forgeyard 的 CA。** Forgeyard 用自己的 CA 提供 HTTPS。從伺服器的 *Trust* 頁面(或
   `https://<forgeyard-host>:7860/api/tls/ca.pem`)下載,並核對頁面上顯示的指紋。放在 Folio 讀得到的地方,例如
   `/etc/folio/forgeyard-ca.pem`。
4. **設定 `.env`**(機密與每台機器各自的值都放這裡;`config/models.yaml` 已經會讀這些變數):

   ```bash
   ASR_BASE_URL=https://<forgeyard-host>:7860/v1
   ASR_API_KEY=<第 2 步的金鑰>
   ASR_MODEL=whisper-large-v3-turbo
   ASR_CA_FILE=/etc/folio/forgeyard-ca.pem
   ```

5. **檢查 `config/models.yaml`**:`rag.asr.provider` 為 `openai-compatible`(或不寫),`enabled: true`。
   重新啟動 Folio。
6. **測試。** 在主控台的 *Settings → Speech* 按 **Test**:它會用已存的設定送出半秒靜音並顯示結果。再上傳一段短錄音
   到資料夾,搜尋其中一句話。

`ASR_CA_FILE` 是**加進**原本的信任清單,所以其他模型呼叫(例如公有雲 API)照常運作。改設 `SSL_CERT_FILE`
則會取代 Folio 所有 HTTPS 呼叫的信任清單。CA 檔不存在或讀不到時,轉錄會失敗並顯示檔案路徑;Folio 絕不會退回不驗證的
TLS。若 Forgeyard 的憑證來自系統已信任的 CA,`ASR_CA_FILE` 留空即可。

**Docker:** 以唯讀方式掛載 CA 檔(`- ./forgeyard-ca.pem:/etc/folio/forgeyard-ca.pem:ro`),並在 `.env`
設定同樣的變數,再加上 `ASR_ENABLED=true`。公開的映像檔沒有 `ffmpeg`,只能送出 24 MB 以內的音訊;較長的錄音請用
`--build-arg INSTALL_FFMPEG=true` 自行建置映像檔([部署 Folio](deploy.zh-TW.md))。

### 其他端點

任何以 `{"text": …}` 回應 OpenAI `POST /v1/audio/transcriptions` 的伺服器都可以:

| 伺服器 | `ASR_BASE_URL` | `ASR_MODEL` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `whisper-1`(或 `gpt-4o-transcribe`) |
| speaches / faster-whisper-server | `http://<host>:8000/v1` | 它提供的模型,例如 `deepdml/faster-whisper-large-v3-turbo-ct2` |
| vLLM(Whisper) | `http://<host>:8000/v1` | 服務的模型名稱,例如 `openai/whisper-large-v3-turbo` |

在 `rag.profile: cloud` 下,金鑰必須是 `${ENV}` 參照,和其他角色一樣([雲端 profile](configuration.zh-TW.md#雲端-profile))。

### Folio 在端點之外仍會做的事

- **長音訊**先重新取樣成 16 kHz 單聲道,再在靜音處切段上傳(`rag.asr.segment_seconds`,預設 600 秒,
  所以每個請求都不會超過 25 MB 的上傳上限),一次兩段。每段遇到 429、5xx 與逾時會重試;有語音卻回傳空白、或疑似
  生成迴圈時會重新請求。這需要 `ffmpeg`。
- **先剪掉開頭的靜音**,讓 whisper 的語言偵測聽到語音。
- **過濾 whisper 常見的幻覺句**(字幕製作名單、「訂閱」片尾、重複的句子)。
- **繁體中文。** whisper 以簡體字輸出中文。`rag.asr.chinese_conversion: s2twp`(預設)會把轉錄結果轉成台灣用語的繁體
  (OpenCC `s2twp`,與 FireRedASR provider 用的轉換相同);`none` 保留端點原本的輸出。英文與其他沒有中文字的
  文字一律不變。
- **清楚的失敗訊息。** 金鑰錯誤(HTTP 401 或 403)時,檔案會以「check rag.asr.api_key」失敗;端點連不上時,
  訊息會寫出端點位址。

### 設定

| 設定 | 變數 | 預設 | |
|---|---|---|---|
| `rag.asr.enabled` | `ASR_ENABLED`(僅 Docker 設定) | `true`(Docker 映像檔為 `false`) | 關閉 = 拒絕上傳音訊。 |
| `rag.asr.provider` | — | `openai-compatible` | 唯一的值。 |
| `rag.asr.base_url` | `ASR_BASE_URL` | 空 | 通常以 `/v1` 結尾。 |
| `rag.asr.api_key` | `ASR_API_KEY` | 空 | 以 `Authorization: Bearer` 送出。 |
| `rag.asr.model` | `ASR_MODEL` | 空 | 以 `model` 送出。 |
| `rag.asr.language` | — | 空 | ISO-639-1 提示(`zh`、`en`);留空由模型自行偵測。 |
| `rag.asr.chinese_conversion` | — | `s2twp` | `s2twp` 或 `none`。 |
| `rag.asr.ca_file` | `ASR_CA_FILE` | 空 | 端點額外信任的 CA(PEM)。 |
| `rag.asr.segment_seconds` / `timeout` / `max_retries` | — | `600` / `300` / `3` | 每段長度、每段逾時、重試次數。 |

除了 `ca_file`,其他設定也能在主控台設定頁修改(需要 `rag:admin`)。

### 呼叫 `POST /v1/transcriptions/general` 的程式

這個端點會回 **410 Gone** 並附上本指南的連結。請改為直接呼叫轉錄端點,使用和 OpenAI 相同的 multipart 表單:

```bash
curl --cacert forgeyard-ca.pem https://<forgeyard-host>:7860/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F model=whisper-large-v3-turbo -F file=@meeting.mp3
```
