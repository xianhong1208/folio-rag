# Upgrading to 1.5: OCR, layout and speech models move out of Folio

[繁體中文](migrate-1.5.zh-TW.md)

Folio 1.5 runs no OCR, layout, table or speech-to-text model of its own. Two things changed:

- **PDFs and images** are read by an **OpenAI-compatible vision model** when you configure one
  (`rag.parsing.vision`, for example Qwen3-VL on Forgeyard). Without one, PDFs are read from their
  **text layer** and image uploads are refused. Docling's in-process layout, TableFormer and
  RapidOCR models are gone, with `rag.docling`, `rag.parsing.images`, the `local` / `cloud` values
  of `rag.parsing.pdf`, and `POST /v1/ocr/general` and `GET /v1/models`.
- **Audio** is transcribed by an OpenAI-compatible endpoint
  ([Speech-to-text](#speech-to-text) below).

Docling itself stays for Office, OpenDocument, HTML, Markdown, e-mail and e-book files: those
backends need no model.

## PDFs and images

Why: on 8 pages of the public eval PDFs (clean and degraded scans, Chinese and English),
Qwen3-VL-8B served by Forgeyard transcribed every page with no misread character and kept the
tables as Markdown tables, while docling's RapidOCR fell to a character F1 of 0.935–0.970 on the
degraded Chinese pages. A vision model also reads tables, charts and handwriting that OCR turns
into loose words, and it takes the GPU memory, the model files and torch's vision stack out of
Folio.

### How a file is read now

| | A vision model is configured | No vision model (the default) |
|---|---|---|
| PDF | Every page is rendered and transcribed to Markdown (tables as tables), the pages joined into one document | Read from the PDF's **text layer**. Prose keeps its paragraphs; a table becomes plain text, row by row |
| PDF page without a text layer (a scan) | Transcribed like any other page | Indexed as a visible marker, `[Page 3: no text layer (a scanned or image-only page); set rag.parsing.vision … to read it.]`; the file shows a warning naming the pages |
| PDF where no page has text | Transcribed | **Indexing fails** with a message saying to configure `rag.parsing.vision`. It is never indexed empty |
| Image (PNG, JPEG, TIFF, WebP, BMP, GIF) | Transcribed | **Upload refused** with 422 and the reason (REST, batch upload's `failed_uploads`, console). An image stored earlier fails indexing with the same message |
| Office, Markdown, HTML, … | docling, as before | docling, as before |

`rag.parsing.pdf` chooses for PDFs:

| Value | PDFs are read by |
|---|---|
| `auto` (default) | the vision model when `rag.parsing.vision` is configured, the text layer otherwise |
| `vision` | always the vision model; Folio refuses to start without one |
| `text` | always the text layer, even with a vision model (images still use the model) |

A vision model counts as configured when it has a `model` and a `base_url` (provider `openai`
needs no `base_url`). Every profile works the same way: `local`, `hybrid` and `cloud` all read
PDFs from their text layer until a vision model is set.

### What happens after the upgrade

| Your configuration | What Folio does |
|---|---|
| A `rag.docling` section in `config/models.yaml` (`device`, `ocr_enabled`, `ocr_model_scale`, `hf_offline`, `compile_models`) | **Refuses to start**, naming this guide. Delete the section. If you relied on `hf_offline: true`, set `HF_HUB_OFFLINE=1` in `.env`. |
| `rag.parsing.pdf: local` or `cloud`, or any `rag.parsing.images` | **Refuses to start**, naming this guide. Remove those lines (`pdf` defaults to `auto`). |
| `rag.parsing.pdf: vision` without a vision model | Refuses to start (and the console refuses the change) |
| The same values stored on the console's Settings page | Deleted at startup with a warning; `rag.parsing` then follows `config/models.yaml`. |
| `DOCLING_DEVICE` in `.env` | Ignored; delete it. |
| `scripts/split_config.py` converting a 1.4 `config.yaml` | Drops `rag.docling`, `rag.parsing.images` and `pdf: local`/`cloud`, and prints a note. |

PDFs and images indexed before the upgrade keep their chunks. Re-index a folder to read its PDFs
the new way.

### Point Folio at a vision model on Forgeyard

1. **Deploy the model.** In Forgeyard's Model library, deploy **Qwen3-VL-8B-Instruct** (vLLM) and name the deployment, for example
   `qwen3-vl-8b`: that name is the model Folio asks for. A smaller card can run Qwen3-VL-4B or
   -2B (less accurate on dense tables).
2. **Create an API key for Folio** on Forgeyard's *API keys* page, limited to that one deployment.
   Copy it once.
3. **Get Forgeyard's CA** from its *Trust* page (or `https://<forgeyard-host>:7860/api/tls/ca.pem`),
   check the fingerprint, and store it where Folio can read it, for example
   `/etc/folio/forgeyard-ca.pem`.
4. **Set `.env`** (`config/models.yaml` already reads these variables):

   ```bash
   VISION_BASE_URL=https://<forgeyard-host>:7860/v1
   VISION_API_KEY=<the key from step 2>
   VISION_MODEL=qwen3-vl-8b
   VISION_CA_FILE=/etc/folio/forgeyard-ca.pem
   ```

5. **Restart Folio.** The startup log says
   `Document parsing (vision) endpoint https://… — openai-compatible / qwen3-vl-8b [pdf=vision, images=vision]`.
6. **Test it.** On *Settings → Indexing*, press **Test** under Document Parsing (it sends a tiny
   image with the saved settings). Then upload a scanned PDF and search for a phrase from it.

`VISION_CA_FILE` is **added** to the usual trust store, like `ASR_CA_FILE`. A wrong key fails the
file with "check rag.parsing.vision base_url / model / api_key"; an untrusted certificate fails it
at once naming `rag.parsing.vision.ca_file`; an endpoint that cannot be reached fails it with its
address. None of them leaves an empty index.

**Docker:** mount the CA read-only (`- ./forgeyard-ca.pem:/etc/folio/forgeyard-ca.pem:ro`) and
set the same four variables in `.env`.

**Pacing.** Up to `rag.parsing.max_concurrency` pages (default 4) are in flight at once across
the whole server, each
with the end of the previous page so tables and paragraphs that cross a page break stay whole.
`max_pages_per_file` (500) and `max_pages_per_day` (unlimited) cap the cost; `dpi` (200) sets
the render resolution. A Forgeyard key may limit images per request and image tokens: Folio sends
one page per request, at most 2048 px on its longer side.

### Settings

| Setting | Variable | Default | |
|---|---|---|---|
| `rag.parsing.pdf` | — | `auto` | `auto`, `vision` or `text` (above). |
| `rag.parsing.vision.provider` | — | `openai-compatible` | Also `openai`, `vllm`, `ollama`, `azure`. |
| `rag.parsing.vision.model` | `VISION_MODEL` | empty | The served model name. |
| `rag.parsing.vision.base_url` | `VISION_BASE_URL` | empty | Usually ends in `/v1`. |
| `rag.parsing.vision.api_key` | `VISION_API_KEY` | empty | Sent as `Authorization: Bearer`. |
| `rag.parsing.vision.ca_file` | `VISION_CA_FILE` | empty | Extra CA (PEM) for the endpoint. |
| `rag.parsing.max_concurrency` / `dpi` / `context_chars` | — | `4` / `200` / `800` | Pages in flight, render resolution, previous-page context. |
| `rag.parsing.max_pages_per_file` / `max_pages_per_day` | — | `500` / `0` (unlimited) | Cost caps. |
| `rag.parsing.on_page_failure` / `min_success_ratio` | — | `mark` / `0.5` | A page that still fails: marked, or the whole file fails. |

Everything except `ca_file`, `dpi`, `context_chars` and `min_success_ratio` can also be changed on
the console's Settings page (needs `rag:admin`).

Any server that answers OpenAI's `POST /v1/chat/completions` with an `image_url` part works:
vLLM (`Qwen/Qwen3-VL-8B-Instruct`), Ollama (`qwen2.5vl`), OpenAI (`gpt-4o-mini`, provider
`openai`) or Azure OpenAI (provider `azure`).

### Callers of `POST /v1/ocr/general` and `GET /v1/models`

Both answer **410 Gone** with a link to this guide. Send the page image to the vision model
directly:

```bash
IMG=$(base64 -w0 page.png)
curl --cacert forgeyard-ca.pem https://<forgeyard-host>:7860/v1/chat/completions \
  -H "Authorization: Bearer $VISION_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl-8b", "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe all text on this page as Markdown."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}]}'
```

## Speech-to-text

Folio 1.5 no longer runs a speech-to-text model itself. The in-process providers
`fireredasr` and `docling-whisper` are removed, with their dependencies (FireRedASR, Whisper,
kaldi) and the `POST /v1/transcriptions/general` endpoint. Audio files are transcribed by an
**OpenAI-compatible endpoint**, `POST {base_url}/audio/transcriptions`, which Forgeyard, speaches,
vLLM, faster-whisper-server and OpenAI all serve.

Why: on Google FLEURS (100 Chinese and 100 English clips), whisper-large-v3-turbo served by
Forgeyard beat the built-in FireRedASR-AED-L: English word error rate 5.8 % against 11.1 %,
Chinese character error rate 9.1 % against 12.0 % (the same on purely Chinese clips). Letting
whisper detect the language was as good as giving it a language hint. The endpoint also takes the
GPU memory, the model files and torch's audio stack out of Folio.

### What happens after the upgrade

| Your configuration | What Folio does |
|---|---|
| `rag.asr.provider: fireredasr` or `docling-whisper` in `config/models.yaml` | **Refuses to start**, with a message naming this guide. Edit the file as below. |
| The same value stored on the console's Settings page | The stored setting is deleted at startup with a warning; `rag.asr` then follows `config/models.yaml`. |
| `rag.asr` with no `base_url` (the default) | Starts. **Audio uploads are refused** with 422 and the reason; Health shows Speech-to-Text as *Not set*. Other files work as before. |
| `rag.asr.enabled: false` | Starts; audio uploads are refused (422). |
| `scripts/split_config.py` converting a 1.4 `config.yaml` | Rewrites the old provider to `openai-compatible` and prints a note; then set the variables below. |

Audio files indexed before the upgrade keep their chunks. A stored audio file that is indexed
again while no endpoint is set fails with the same message, and is never indexed empty.

### Point Folio at Forgeyard

1. **Deploy the model.** In Forgeyard, deploy *Whisper large-v3 turbo (faster-whisper)* from
   the Model library (it runs on the speaches engine, about 3 GiB of GPU memory, or on a CPU).
   Name the deployment, for example `whisper-large-v3-turbo`: that name is the model Folio asks for.
2. **Create an API key for Folio** on Forgeyard's *API keys* page and limit it to that one
   deployment, so the key cannot reach your other models. Copy it once.
3. **Get Forgeyard's CA.** Forgeyard serves HTTPS with its own CA. Download it from the
   server's *Trust* page (or `https://<forgeyard-host>:7860/api/tls/ca.pem`) and check the
   fingerprint the page shows. Store it where Folio can read it, for example
   `/etc/folio/forgeyard-ca.pem`.
4. **Set `.env`** (secrets and per-machine values live there; `config/models.yaml` already reads
   these variables):

   ```bash
   ASR_BASE_URL=https://<forgeyard-host>:7860/v1
   ASR_API_KEY=<the key from step 2>
   ASR_MODEL=whisper-large-v3-turbo
   ASR_CA_FILE=/etc/folio/forgeyard-ca.pem
   ```

5. **Check `config/models.yaml`**: `rag.asr.provider` is `openai-compatible` (or left out),
   `enabled: true`. Restart Folio.
6. **Test it.** On the console's *Settings → Speech*, press **Test**: it sends half a second of
   silence with the saved settings and shows the answer. Then upload a short recording to a
   folder and search for a phrase from it.

`ASR_CA_FILE` is **added** to the usual trust store, so every other model call (a public cloud
API, for example) keeps working. Setting `SSL_CERT_FILE` instead would replace the trust store
for every HTTPS call Folio makes. A CA file that is missing or unreadable fails the transcription
with its path; Folio never falls back to unverified TLS. If Forgeyard's certificate comes from a
CA your system already trusts, leave `ASR_CA_FILE` empty.

**Docker:** mount the CA file read-only (`- ./forgeyard-ca.pem:/etc/folio/forgeyard-ca.pem:ro`)
and set the same variables plus `ASR_ENABLED=true` in `.env`. The published image has no
`ffmpeg`, so only audio up to 24 MB can be sent; for longer recordings build the image with
`--build-arg INSTALL_FFMPEG=true` ([Deploying Folio](deploy.md)).

### Other endpoints

Anything that answers OpenAI's `POST /v1/audio/transcriptions` with `{"text": …}` works:

| Server | `ASR_BASE_URL` | `ASR_MODEL` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `whisper-1` (or `gpt-4o-transcribe`) |
| speaches / faster-whisper-server | `http://<host>:8000/v1` | the model it serves, e.g. `deepdml/faster-whisper-large-v3-turbo-ct2` |
| vLLM (Whisper) | `http://<host>:8000/v1` | the served model name, e.g. `openai/whisper-large-v3-turbo` |

Under `rag.profile: cloud` the key must be a `${ENV}` reference, as for the other roles
([Cloud profile](configuration.md#cloud-profile)).

### What Folio still does around the endpoint

- **Long audio** is resampled to 16 kHz mono and uploaded in pieces cut at silences
  (`rag.asr.segment_seconds`, 600 s by default, so no request passes a 25 MB upload limit), two
  pieces at a time. Each piece is retried on 429, 5xx and timeouts, and asked again when it comes
  back empty although it holds speech, or looping. This needs `ffmpeg`.
- **Leading silence is trimmed** first, so whisper's language detection hears speech.
- **Whisper's stock hallucinations** (subtitle credits, "subscribe" outros, repeated phrases) are
  filtered out.
- **Traditional Chinese.** Whisper writes Chinese in Simplified characters. With
  `rag.asr.chinese_conversion: s2twp` (the default) Folio converts the transcript to Traditional
  Chinese with Taiwan phrasing (OpenCC `s2twp`, the conversion the FireRedASR provider used);
  `none` keeps what the endpoint returns. English and other text without Chinese characters is
  never changed.
- **Clear failures.** A wrong key (HTTP 401 or 403) fails the file with "check rag.asr.api_key";
  an unreachable endpoint fails it with the endpoint's address.

### Settings

| Setting | Variable | Default | |
|---|---|---|---|
| `rag.asr.enabled` | `ASR_ENABLED` (Docker config only) | `true` (`false` in the Docker image) | Off = audio uploads are refused. |
| `rag.asr.provider` | — | `openai-compatible` | The only value. |
| `rag.asr.base_url` | `ASR_BASE_URL` | empty | Usually ends in `/v1`. |
| `rag.asr.api_key` | `ASR_API_KEY` | empty | Sent as `Authorization: Bearer`. |
| `rag.asr.model` | `ASR_MODEL` | empty | Sent as `model`. |
| `rag.asr.language` | — | empty | ISO-639-1 hint (`zh`, `en`); empty lets the model detect it. |
| `rag.asr.chinese_conversion` | — | `s2twp` | `s2twp` or `none`. |
| `rag.asr.ca_file` | `ASR_CA_FILE` | empty | Extra CA (PEM) for the endpoint. |
| `rag.asr.segment_seconds` / `timeout` / `max_retries` | — | `600` / `300` / `3` | Piece length, per-piece timeout, retries. |

Everything except `ca_file` can also be changed on the console's Settings page (needs
`rag:admin`).

### Callers of `POST /v1/transcriptions/general`

The endpoint answers **410 Gone** with a link to this guide. Call the transcription endpoint
directly instead, with the same multipart form OpenAI uses:

```bash
curl --cacert forgeyard-ca.pem https://<forgeyard-host>:7860/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F model=whisper-large-v3-turbo -F file=@meeting.mp3
```
