# 驗證、資料夾歸屬與撤銷

[English](auth.md)

## 資料夾歸屬

資料夾屬於**一個人** —— 也就是 token 所發給的那位 MCP Gate 使用者 —— 而不是某一把 token。擁有者 key 是 token 的 `sub` claim,存成 `sub:<sub>`(`Folders.user_token`;檔案放在 `storage/sub_<sub>/<folder>/`)。MCP Gate 在 `sub` 放的是:

| Token | `sub` | 誰看得到這些資料夾 |
|-------|-------|--------------------|
| Personal access token | 使用者的 id | 該使用者的每一把 PAT 與每一次 OAuth 登入 |
| OAuth access token(authorization code、refresh) | 同一個使用者 id | 同上 —— 每 15 分鐘換一把新的 access token,資料夾仍在 |
| `client_credentials` token | 該 client 自己的 `client_id` | 只有那個 client(機器身分,不是使用者) |

沒有 `sub` 的 token 不能擁有資料夾:REST 回 401,MCP 工具呼叫回 `unauthorized`,工具清單不顯示任何資料夾工具,主控台拒絕登入(`missing_sub`)。`auth.enabled: false`(開發模式)維持不變:主控台的資料夾擁有者是 `console`。

在 MCP Gate 撤銷一把 token,停用的是那把 token,而不是資料夾:資料夾屬於那個人,用他的其他 token 仍然存取得到。要讓某人失去資料夾,請刪除資料夾,或在 MCP Gate 停用該使用者。

**1.4.1 之前的資料夾**以建立它的 token 的 `jti` 為 key。它們不會被整批遷移 —— Folio 這邊沒有任何資料能說明某個 `jti` 屬於哪個人。改成:帶著這種 token 的第一個通過驗證的請求(REST、`/mcp` 工具清單或呼叫、主控台登入或 session)會把該 token 擁有的所有資料夾移到它的 `sub` 名下:資料列、儲存目錄與檔案記錄的路徑、`Folio_<folder>` 工具,在同一個交易內完成,而且只做一次。同一使用者的並行請求會被序列化(行程內鎖加上 PostgreSQL advisory lock,資料列以 `FOR UPDATE` 鎖定),所以不會重複搬移,也不會只搬一半。如果使用者已經有同名資料夾(他的兩把 PAT 各有一個 `docs`),被移過來的那個會改用下一個可用名稱 —— `docs-2`、`docs-3`…… —— 並在設定 **Audit Log** 記下這次改名;兩個資料夾絕不會合併。token 再也沒被使用的資料夾(已撤銷或過期)會留在原處;管理者可在 Overview 看到還剩多少(`GET /api/admin/overview` 的 `legacy_owned_folders`)。

## Token 撤銷

Folio 以離線方式驗證 MCP Gate 的 token,因此預設情況下,在 MCP Gate 撤銷的 token 在這裡會一直有效到過期為止——而 PAT 最長可活一年。啟用 **revocation feed** 後,Folio 會在背景輪詢 MCP Gate 的 `POST /oauth/revoked`(每 `poll_interval_sec` 秒,帶 jitter 與錯誤退避),在記憶體保存「已撤銷但尚未過期」的 `jti`,並在所有路徑拒絕它們:`/mcp`(401 `invalid_token`)、REST API(401)、主控台登入,以及既有主控台 session 的每一個請求(session 以 401 `token_revoked` 結束)。撤銷約在一個輪詢間隔內生效。

| 設定 | 預設 | 用途 |
|------|------|------|
| `auth.revocation.enabled` | `false`(`FOLIO_REVOCATION_ENABLED`) | 開啟 feed。關閉 = 舊行為,不輪詢。 |
| `auth.revocation.client_id` | `${FOLIO_REVOCATION_CLIENT_ID}` | 在 MCP Gate 註冊的 resource-server client(見下方)。 |
| `auth.revocation.client_secret` | `${FOLIO_REVOCATION_CLIENT_SECRET}` | 它的 secret。**必須**是 `${ENV_VAR}` 參照——寫成字面值會拒絕啟動。 |
| `auth.revocation.poll_interval_sec` | `30` | 輪詢頻率(5–3600 秒)。 |
| `auth.revocation.fail_mode` | `open` | MCP Gate 連不上時:`open` 繼續接受 token 並大聲記錄;`closed` 在連續 `stale_after_intervals` × `poll_interval_sec` 沒有成功輪詢後拒絕所有 token(`/mcp` 與 REST 回 401,主控台回 503),啟動後第一次輪詢成功前也一樣。 |
| `auth.revocation.stale_after_intervals` | `4` | 清單何時算過期(`/health/detailed` 顯示 `degraded`;`closed` 開始拒絕)。 |

第一次輪詢抓完整清單;之後只問上次以來的撤銷(重疊 30 秒),每小時再完整抓一次。token 本來就會過期之後,項目即被移除。啟動時會記錄 feed 是否啟用,`GET /health/detailed` 的 `checks.revocation` 會顯示 `last_success`、`revoked_count`、`consecutive_failures`、`last_error`、`stale`。

**設定步驟**(每個部署做一次):

1. **在 MCP Gate 註冊 confidential client。** 主控台 → *OAuth Clients* → *Register trusted client*。名稱例如 `Folio revocation feed`;*Redirect URIs* 留空;*Grant types* 只選 `client_credentials`;*Token endpoint authentication*:**Client secret (POST body)**(HTTP Basic 也可以,MCP Gate 兩者都接受);*Resource server for*:**Folio**。註冊後複製 `client_id`(`mcpc_…`)與 client secret——secret 只顯示一次。
2. **綁定到 Folio**——若步驟 1 已選 *Resource server for* 就已完成。否則在該 client 那一列點伺服器圖示(*Resource server…*),選 Folio 服務後 *Save*;或以主控台管理員呼叫 `PUT /api/oauth/clients/<client_id>/resource-server`,body `{"service_id": "<Folio service id>"}`。該列會顯示 *Resource server for Folio*。feed 只列出 audience 等於該服務 audience 的 token,所以它必須等於 Folio 的 `auth.audience`。
3. **把憑證交給 Folio**(環境變數或 `.env`),然後重啟:
   ```bash
   FOLIO_REVOCATION_ENABLED=true
   FOLIO_REVOCATION_CLIENT_ID=mcpc_...
   FOLIO_REVOCATION_CLIENT_SECRET=...
   ```
   (`config/auth.yaml` 會讀這三個變數;`.env.example` 有列出。)
4. **確認。** 啟動 log 會出現 `MCP Gate revocation feed: ENABLED …`;`curl -s localhost:5039/health/detailed | jq .checks.revocation` 會看到最近的 `last_success`。在 MCP Gate 撤銷一個測試 PAT,`poll_interval_sec` 內用它發請求會得到 401。`last_error` 出現 `HTTP 401: invalid_client` 表示憑證錯誤;`HTTP 403: unauthorized_client` 表示 client 尚未綁定伺服器。
