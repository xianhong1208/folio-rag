# Authentication, folder ownership and revocation

[繁體中文](auth.zh-TW.md)

## Folder ownership

A folder belongs to a **person** — the MCP Gate user its token was issued to — not to one token. The owner key is the token's `sub` claim, stored as `sub:<sub>` (`Folders.user_token`; files live under `storage/sub_<sub>/<folder>/`). What MCP Gate puts in `sub`:

| Token | `sub` | Who sees the folders |
|-------|-------|----------------------|
| Personal access token | the user's id | every PAT and OAuth sign-in of that user |
| OAuth access token (authorization code, refresh) | the same user id | the same — a new access token every 15 minutes keeps its folders |
| `client_credentials` token | the client's own `client_id` | that client only (a machine identity, never a user) |

A token without a `sub` cannot own folders: REST answers 401, an MCP tool call returns `unauthorized`, the tool list shows no folder tools, and the console refuses the sign-in (`missing_sub`). With `auth.enabled: false` (dev mode) nothing changes: console folders are owned by `console`.

Revoking a token in MCP Gate stops that token, not the folders: they belong to the person and stay reachable with their other tokens. To take folders away from someone, delete them or disable the user in MCP Gate.

**Folders from before 1.4.1** were keyed on the creating token's `jti`. They are not migrated in bulk — nothing on the Folio side says which person a `jti` belonged to. Instead, the first verified request with such a token (REST, `/mcp` tool list or call, console sign-in or session) moves every folder that token owns to its `sub`: the rows, the storage directory and the files' stored paths, and the `Folio_<folder>` tools, in one transaction, exactly once. Concurrent requests are serialised per user (an in-process lock plus a PostgreSQL advisory lock, with the rows locked `FOR UPDATE`), so nothing is moved twice or half-moved. If the user already has a folder of that name (two of their PATs each had a `docs`), the moved one gets the next free name — `docs-2`, `docs-3`, … — and the settings **Audit Log** records the rename; two folders are never merged. Folders whose token is never used again (revoked or expired) stay where they are; admins see how many are left on the Overview (`legacy_owned_folders` in `GET /api/admin/overview`).

## Token revocation

Folio verifies MCP Gate tokens offline, so by default a token revoked in MCP Gate keeps working here until it expires — and a PAT can live up to a year. Enable the **revocation feed** and Folio polls MCP Gate's `POST /oauth/revoked` in the background (every `poll_interval_sec`, with jitter and error backoff), keeps the revoked-but-unexpired `jti`s in memory, and rejects them everywhere: `/mcp` (401 `invalid_token`), the REST API (401), console sign-in, and every request of an existing console session (the session ends with 401 `token_revoked`). A revocation takes effect within about one poll interval.

| Setting | Default | Purpose |
|---------|---------|---------|
| `auth.revocation.enabled` | `false` (`FOLIO_REVOCATION_ENABLED`) | Turn the feed on. Off = the old behaviour, nothing is polled. |
| `auth.revocation.client_id` | `${FOLIO_REVOCATION_CLIENT_ID}` | The resource-server client registered in MCP Gate (below). |
| `auth.revocation.client_secret` | `${FOLIO_REVOCATION_CLIENT_SECRET}` | Its secret. **Must** be an `${ENV_VAR}` reference — a literal value refuses to start. |
| `auth.revocation.poll_interval_sec` | `30` | How often the feed is polled (5–3600). |
| `auth.revocation.fail_mode` | `open` | When MCP Gate is unreachable: `open` keeps accepting tokens and logs loudly; `closed` rejects every token (401 on `/mcp` and REST, 503 in the console) once no poll has succeeded for `stale_after_intervals` × `poll_interval_sec`, and until the first poll after startup succeeds. |
| `auth.revocation.stale_after_intervals` | `4` | When the list counts as stale (`/health/detailed` reports `degraded`; `closed` starts rejecting). |

The first poll fetches the whole list; later polls ask only for revocations since the previous one (with a 30 s overlap), and a full fetch repeats every hour. Entries are dropped once the token would have expired anyway. Startup logs whether the feed is enabled, and `GET /health/detailed` shows it under `checks.revocation` (`last_success`, `revoked_count`, `consecutive_failures`, `last_error`, `stale`).

**Setting it up** (once per deployment):

1. **Register a confidential client in MCP Gate.** Console → *OAuth Clients* → *Register trusted client*. Name it e.g. `Folio revocation feed`; leave *Redirect URIs* empty; under *Grant types* select only `client_credentials`; *Token endpoint authentication*: **Client secret (POST body)** (HTTP Basic works too — MCP Gate accepts either); *Resource server for*: **Folio**. Register, then copy the `client_id` (`mcpc_…`) and the client secret — the secret is shown once.
2. **Bind it to Folio** — already done if you picked *Resource server for* in step 1. Otherwise, on the client's row click the server icon (*Resource server…*), choose the Folio service and *Save*; or call `PUT /api/oauth/clients/<client_id>/resource-server` with `{"service_id": "<Folio service id>"}` as a console admin. The row then reads *Resource server for Folio*. The feed lists only tokens whose audience is that service's audience, so it must equal Folio's `auth.audience`.
3. **Give Folio the credentials** in its environment (or `.env`), then restart:
   ```bash
   FOLIO_REVOCATION_ENABLED=true
   FOLIO_REVOCATION_CLIENT_ID=mcpc_...
   FOLIO_REVOCATION_CLIENT_SECRET=...
   ```
   (`config/auth.yaml` reads these three variables; `.env.example` lists them.)
4. **Check it.** The startup log says `MCP Gate revocation feed: ENABLED …`; `curl -s localhost:5039/health/detailed | jq .checks.revocation` shows a recent `last_success`. Revoke a test PAT in MCP Gate; within `poll_interval_sec` a request with it gets 401. A `HTTP 401: invalid_client` in `last_error` means wrong credentials; `HTTP 403: unauthorized_client` means the client is not bound to a server.
