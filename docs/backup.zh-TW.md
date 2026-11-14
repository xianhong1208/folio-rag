# 備份與還原

[English](backup.md)

`main.py backup` 會寫出一個封存檔,裡面有讓一個 Folio 完整回來所需的一切:資料庫(每張表,包括每個
資料夾的 pgvector 表 `data_<id>_<uuid>` 與其中的向量和索引)、`storage/` 底下的檔案,以及一份
manifest。`main.py restore` 則把封存檔還原到空的資料庫與 `storage/`。伺服器也可以依排程自動備份。

## 封存檔裡有什麼

`folio-backup-<UTC 時間>.tar` 是一般的 tar 檔:

| 項目 | 內容 |
|---|---|
| `database.dump` | Folio 資料庫的 `pg_dump --format=custom`(已壓縮) |
| `storage/…` | `storage/` 底下的每個檔案:上傳的檔案與主控台的 session 密鑰 |
| `manifest.json` | 封存檔的來源資訊(見下),以及其他每個項目的 SHA-256 |

manifest 記錄了 Folio 版本、資料庫 migration revision、embedding 模型與維度(以及是否被設定頁面
覆寫)、關鍵字搜尋用的斷詞器、已索引檔案實際使用的 embedding 模型、每個資料夾的檔案數與 chunk
數、當時正在跑的索引工作、資料庫有紀錄但檔案已不存在的項目,以及 `pg_dump` 與伺服器的版本。

封存檔含有所有文件與主控台用來簽 session 的密鑰,請把它當成資料庫本身一樣保管。

## 需求

- PostgreSQL 用戶端工具 `pg_dump` 與 `pg_restore`,**主版本要和伺服器相同或更新**(例如
  PostgreSQL 15–17 的伺服器用 `apt install postgresql-client-17`)。不在 `PATH` 上的話,設定
  `database.backup.pg_bin_dir`(例如 `/usr/lib/postgresql/17/bin`)。還原時用的 `pg_restore`
  不能比做出封存檔的 `pg_dump` 舊(manifest 有寫是哪一版)。
- 不需要 superuser。`DATABASE_URL` 的角色要擁有 Folio 的資料表(由 Folio 自己建立的表就是)。
  要還原到新的資料庫,角色需要 `CREATEDB`,而且新資料庫裡要有 `vector` extension:pgvector 不是
  trusted extension,所以角色不是 superuser 時,要先請管理員在新資料庫執行
  `CREATE EXTENSION vector;`。缺少時,還原會直接告訴你。
- 用 Docker 時,映像檔裡要有用戶端工具,備份目錄也要像 `storage/` 一樣掛成 volume。

## 備份

```bash
uv run python main.py backup                       # 寫到 database.backup.dir(預設 backups/)
uv run python main.py backup --keep 7              # 寫完後只保留該目錄最新的 7 份
uv run python main.py backup --output /srv/b/folio.tar
```

| 選項 | 意思 |
|---|---|
| `--dir DIR` | 把 `folio-backup-<UTC 時間>.tar` 寫到 `DIR`,而不是 `database.backup.dir` |
| `--output FILE` | 就寫這個檔案(已存在的檔案絕不覆蓋) |
| `--keep N` | 寫完後,該目錄只保留最新的 `N` 份帶時間戳記的封存檔 |
| `--allow-running-jobs` | 即使有索引工作在跑也照樣備份(見下) |
| `--config-dir DIR` | 設定目錄,和伺服器相同 |

Folio 運作中也可以備份。結束代碼:`0` 已寫出、`1` 失敗(例如找不到 `pg_dump` 或它出錯)、
`2` 拒絕(前置條件不符,最後一行會說明原因)。

**一致性。** 資料庫是從同一個快照 dump 出來的:Folio 開一個 `REPEATABLE READ` 交易、匯出它的
快照交給 `pg_dump --snapshot`,manifest 裡的資料夾清單與已索引模型也從同一個快照讀出。
`storage/` 緊接著封存。Folio 只會新增或刪除儲存的檔案(從不原地改寫),所以封存檔最多多出一個
快照之後才上傳的檔案(無害),或少掉一個期間被刪除的檔案(manifest 會列出)。

**索引工作。** 有索引工作在等待或執行時,備份會被拒絕:正在寫入的 chunk 和檔案的索引紀錄在封存檔
裡會對不上。請等工作結束,或加上 `--allow-running-jobs`;manifest 會列出這些工作,還原後 Folio
啟動時會把它們標成中斷,主控台會對那些檔案提供 **Retry**。紀錄超過 15 分鐘沒有更新的工作,
表示原本的伺服器已經不在,不會擋住備份。

同一個目錄同時只會有一個備份在寫。

## 排程備份

```yaml
# config/database.yaml
database:
  backup:
    dir: "backups"          # 相對路徑 = 在 config/ 與 storage/ 旁邊
    enabled: true           # 伺服器每 interval_hours 小時備份一次
    interval_hours: 24
    keep: 7                 # 保留最新的 7 份帶時間戳記的封存檔
    pg_bin_dir: ""          # pg_dump / pg_restore 所在目錄;空白 = PATH
```

`enabled: true` 時,伺服器在自己的 event loop 上執行備份。下一次備份的時間是 `dir` 裡最新封存檔
(不論是誰做的)的 `interval_hours` 小時之後,所以重啟不會漏掉也不會重複;`dir` 是空的時候,
第一次會在啟動兩分鐘後執行。有索引工作在跑時會等待,每 10 分鐘再試一次;等了 6 小時仍在跑,
就照樣備份並記錄這些工作。每次備份後,超過 `keep` 份的最舊時間戳記封存檔會被刪除。其他名稱的
檔案(`--output` 寫出的、你改過名的副本)絕不會被刪。某一輪失敗會記到 log,一小時後重試。

之所以內建在伺服器而不是交給 cron,是因為這樣在 Docker 和一般主機上行為一致、它知道索引工作
何時在跑,而且狀態會出現在健康檢查裡。如果你偏好 cron(例如要寫到另一台機器掛載的目錄),就維持
`enabled: false`,改用:

```cron
30 3 * * *  cd /opt/folio && uv run python main.py backup --keep 7 >> logs/backup.log 2>&1
```

不論哪種方式,`GET /health/detailed` 都有一個 `backup` 區塊:最新的封存檔(`name`、`created_at`、
`size_bytes`、`age_hours`);排程開啟時還有 `last_attempt_at`、`last_result`(`ok`、`failed`、
`waiting`)、`last_error`(只有錯誤種類,細節在伺服器 log)與 `next_run_at`。某一輪失敗,或最新
封存檔已經超過兩個間隔,這個區塊和整體狀態都會變成 `degraded`。

## 還原

1. 停止 Folio。只要還有其他連線連著資料庫,還原就會拒絕。
2. 先用這個 Folio 檢查封存檔,不會改動任何東西:

   ```bash
   uv run python main.py restore backups/folio-backup-20261114T035002Z.tar --check
   ```

3. 還原:

   ```bash
   uv run python main.py restore backups/folio-backup-20261114T035002Z.tar
   ```

4. 啟動 Folio。封存檔來自較舊的 Folio 時,啟動時的 migration 會像一般升級一樣把資料庫更新到最新。

還原的目標是 `DATABASE_URL` 指向的資料庫(不存在就建立)與 `storage/`(`FOLIO_STORAGE_ROOT`)。
要還原到別處,就只在那一個指令把 `DATABASE_URL` 和 `FOLIO_STORAGE_ROOT` 指過去。

在改動任何東西之前,它會檢查以下各項,不符就拒絕:

| 檢查 | 除非加上 |
|---|---|
| 封存檔本身:每個項目都是一般檔案、名稱安全(沒有絕對路徑、沒有 `..`、沒有連結)、列在 manifest 且 SHA-256 相符,而且 manifest 列的項目一個都不少 | 一律拒絕 |
| 封存檔來自較新的 Folio(不認得的 migration revision 或較高的版本) | `--allow-mismatch` |
| 向量是用和這個 Folio 不同的 embedding 模型或維度做的 | `--allow-mismatch`(之後要重新索引每個資料夾) |
| 資料庫已經有資料表 | `--force`:那些表會被刪除 |
| `storage/` 裡已經有資料夾 | `--force`:會被移到 `storage.before-restore-<時間>`,不會刪除 |
| 還有其他連線連著資料庫 | 一律拒絕:請先停止 Folio |

embedding 模型的判斷:在設定頁面存過的設定會跟著資料庫回來並優先;否則以這個 Folio 的
`config/models.yaml` 為準。斷詞器不同只會警告:還原後請在每個資料夾執行 **Rebuild FTS**。

資料庫在單一交易裡還原(`pg_restore --single-transaction`),失敗時不會留下還原到一半的狀態。
動到資料庫之前,封存檔會先在 `storage/` 旁邊解開並驗證。dump 裡的 `vector` extension 與
`public` schema 項目會略過(前者事先建立,後者屬於資料庫擁有者),所以非 superuser 也能還原。

只還原你信任的封存檔:`pg_restore` 會執行 dump 裡的 SQL。

## 把 Folio 搬到另一台機器

在舊機器備份、複製封存檔、準備新機器(相同或更新的 Folio、相同的 embedding 模型與維度、一個裝好
pgvector 的空資料庫),執行 `main.py restore`,再啟動 Folio。只要新機器驗證的是同一個 MCP Gate
issuer、同一個 audience 的 token,原本的 token 都能繼續用。
