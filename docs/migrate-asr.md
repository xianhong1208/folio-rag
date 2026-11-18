# Moving speech-to-text to an endpoint (1.5)

[繁體中文](migrate-asr.zh-TW.md)

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

## What happens after the upgrade

| Your configuration | What Folio does |
|---|---|
| `rag.asr.provider: fireredasr` or `docling-whisper` in `config/models.yaml` | **Refuses to start**, with a message naming this guide. Edit the file as below. |
| The same value stored on the console's Settings page | The stored setting is deleted at startup with a warning; `rag.asr` then follows `config/models.yaml`. |
| `rag.asr` with no `base_url` (the default) | Starts. **Audio uploads are refused** with 422 and the reason; Health shows Speech-to-Text as *Not set*. Other files work as before. |
| `rag.asr.enabled: false` | Starts; audio uploads are refused (422). |
| `scripts/split_config.py` converting a 1.4 `config.yaml` | Rewrites the old provider to `openai-compatible` and prints a note; then set the variables below. |

Audio files indexed before the upgrade keep their chunks. A stored audio file that is indexed
again while no endpoint is set fails with the same message, and is never indexed empty.

## Point Folio at Forgeyard

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

## Other endpoints

Anything that answers OpenAI's `POST /v1/audio/transcriptions` with `{"text": …}` works:

| Server | `ASR_BASE_URL` | `ASR_MODEL` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `whisper-1` (or `gpt-4o-transcribe`) |
| speaches / faster-whisper-server | `http://<host>:8000/v1` | the model it serves, e.g. `deepdml/faster-whisper-large-v3-turbo-ct2` |
| vLLM (Whisper) | `http://<host>:8000/v1` | the served model name, e.g. `openai/whisper-large-v3-turbo` |

Under `rag.profile: cloud` the key must be a `${ENV}` reference, as for the other roles
([Cloud profile](configuration.md#cloud-profile)).

## What Folio still does around the endpoint

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

## Settings

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

## Callers of `POST /v1/transcriptions/general`

The endpoint answers **410 Gone** with a link to this guide. Call the transcription endpoint
directly instead, with the same multipart form OpenAI uses:

```bash
curl --cacert forgeyard-ca.pem https://<forgeyard-host>:7860/v1/audio/transcriptions \
  -H "Authorization: Bearer $ASR_API_KEY" \
  -F model=whisper-large-v3-turbo -F file=@meeting.mp3
```
