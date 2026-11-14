# Backup and restore

[繁體中文](backup.zh-TW.md)

`main.py backup` writes one archive that holds everything a Folio instance needs to come back:
the database (every table, including each folder's pgvector table `data_<id>_<uuid>` with its
vectors and indexes), the files under `storage/`, and a manifest. `main.py restore` puts an archive
back into an empty database and `storage/`. The server can also make a backup on a schedule.

## What an archive holds

`folio-backup-<UTC time>.tar` is a plain tar file:

| Entry | What it is |
|---|---|
| `database.dump` | `pg_dump --format=custom` of Folio's database (compressed) |
| `storage/…` | every file under `storage/`: the uploaded files and the console session secret |
| `manifest.json` | what the archive was made from (below) and a SHA-256 for every other entry |

The manifest records the Folio version, the database migration revision, the embedding model and
dimension (and whether the Settings page overrode them), the segmenter used for keyword search,
the embedding models the indexed files were really made with, every folder with its file and chunk
counts, any index job that was running, files the database refers to that were missing, and the
`pg_dump` and server versions.

An archive contains every document and the console's session-signing secret. Keep it as private
as the database itself.

## Requirements

- The PostgreSQL client tools `pg_dump` and `pg_restore`, **the same major version as the server
  or newer** (e.g. `apt install postgresql-client-17` for a PostgreSQL 15–17 server). If they are
  not on `PATH`, set `database.backup.pg_bin_dir` (e.g. `/usr/lib/postgresql/17/bin`). Restore with
  a `pg_restore` at least as new as the `pg_dump` that made the archive (the manifest says which).
- No superuser is needed. The role in `DATABASE_URL` must own Folio's tables (it does when Folio
  created them). To restore into a new database, the role needs `CREATEDB`, and the `vector`
  extension must exist in it: pgvector is not a trusted extension, so when the role is not a
  superuser an administrator runs `CREATE EXTENSION vector;` in the new database first. The restore
  says so when it is missing.
- In Docker, the image needs the client tools and the backup directory should be a volume, like
  `storage/`.

## Making a backup

```bash
uv run python main.py backup                       # into database.backup.dir (default backups/)
uv run python main.py backup --keep 7              # then delete all but the newest 7 there
uv run python main.py backup --output /srv/b/folio.tar
```

| Option | Meaning |
|---|---|
| `--dir DIR` | write `folio-backup-<UTC time>.tar` into `DIR` instead of `database.backup.dir` |
| `--output FILE` | write exactly this file (never overwrites an existing one) |
| `--keep N` | afterwards keep only the newest `N` timestamped archives in that directory |
| `--allow-running-jobs` | back up although an index job is running (see below) |
| `--config-dir DIR` | the config directory, as for the server |

It runs while Folio is running. Exit codes: `0` written, `1` failed (e.g. `pg_dump` missing or
failing), `2` refused (a precondition, said on the last line).

**Consistency.** The database is dumped from one snapshot: Folio opens a `REPEATABLE READ`
transaction, exports its snapshot and hands it to `pg_dump --snapshot`, and reads the folder list
and the indexed models for the manifest from the same snapshot. `storage/` is archived right after.
Folio only adds and deletes stored files (it never rewrites one in place), so the archive can at
most hold a file uploaded after the snapshot, which is harmless, or miss one deleted in between,
which the manifest lists.

**Index jobs.** While an index job is pending or running, the backup is refused: the chunks being
written and the file's index record would not match in the archive. Wait for the job, or pass
`--allow-running-jobs`; the manifest then lists the jobs, and after a restore Folio marks them
interrupted at start-up, so the console offers **Retry** for their files. A job whose record has not
moved for 15 minutes belongs to a server that is gone and does not block.

Only one backup writes into a directory at a time.

## Scheduled backups

```yaml
# config/database.yaml
database:
  backup:
    dir: "backups"          # relative = next to config/ and storage/
    enabled: true           # the server makes one every interval_hours
    interval_hours: 24
    keep: 7                 # the newest 7 timestamped archives are kept
    pg_bin_dir: ""          # where pg_dump / pg_restore are; empty = PATH
```

With `enabled: true` the server runs the backups itself, on its own event loop. The next one is due
`interval_hours` after the newest archive in `dir`, whoever made it, so a restart neither skips nor
repeats one; the first comes two minutes after start-up when `dir` is empty. While an index job
runs it waits and tries again every 10 minutes; after 6 hours it backs up anyway and records the
jobs. After each backup the oldest timestamped archives beyond `keep` are deleted. Files with other
names (a `--output` archive, a copy you renamed) are never deleted. A failed round is logged and
retried an hour later.

This is built into the server rather than left to cron because it then works the same in Docker and
on a plain host, it knows when indexing is running, and its state shows in the health check. If you
prefer cron (e.g. to write to another machine's mount), leave `enabled: false` and run:

```cron
30 3 * * *  cd /opt/folio && uv run python main.py backup --keep 7 >> logs/backup.log 2>&1
```

`GET /health/detailed` has a `backup` block either way: the newest archive (`name`, `created_at`,
`size_bytes`, `age_hours`), and with the schedule on `last_attempt_at`, `last_result` (`ok`,
`failed`, `waiting`), `last_error` (the kind of error only; details are in the server log) and
`next_run_at`. A failed round, or a newest archive older than two intervals, turns the block and
the overall status `degraded`.

## Restoring

1. Stop Folio. The restore refuses while any other connection to the database is open.
2. Check the archive against this Folio. Nothing is changed:

   ```bash
   uv run python main.py restore backups/folio-backup-20261114T035002Z.tar --check
   ```

3. Restore:

   ```bash
   uv run python main.py restore backups/folio-backup-20261114T035002Z.tar
   ```

4. Start Folio. If the archive came from an older Folio, the start-up migration brings the
   database up to date, as after any upgrade.

The restore goes to the database in `DATABASE_URL` (created if it does not exist) and to
`storage/` (`FOLIO_STORAGE_ROOT`). To restore somewhere else, point `DATABASE_URL` and
`FOLIO_STORAGE_ROOT` there for that one command.

What it checks, and refuses on, before it changes anything:

| Check | Refused unless |
|---|---|
| The archive: every entry is a regular file with a safe name (no absolute paths, no `..`, no links), listed in the manifest with a matching SHA-256, and nothing listed is missing | never |
| The archive comes from a newer Folio (an unknown migration revision or a higher version) | `--allow-mismatch` |
| The vectors were made with another embedding model or dimension than this Folio uses | `--allow-mismatch` (then re-index every folder) |
| The database already has tables | `--force`: they are dropped |
| `storage/` already holds folders | `--force`: it is moved aside to `storage.before-restore-<time>`, never deleted |
| Another connection to the database is open | never: stop Folio first |

For the embedding model, settings saved on the Settings page come back with the database and win;
otherwise this instance's `config/models.yaml` decides. A different segmenter only warns: run
**Rebuild FTS** on each folder after the restore.

The database is restored in one transaction (`pg_restore --single-transaction`), so a failure leaves
nothing half-restored. The archive is unpacked and verified next to `storage/` before the database
is touched. The `vector` extension and the `public` schema entries of the dump are skipped (the
first is created beforehand, the second belongs to the database owner), which is what lets a
non-superuser restore.

Only restore archives you trust: `pg_restore` replays the SQL inside the dump.

## Moving Folio to another machine

Back up on the old machine, copy the archive, set up the new one (same or newer Folio, the same
embedding model and dimension, an empty database with pgvector), run `main.py restore`, then start
Folio. Tokens keep working if the new instance verifies tokens from the same MCP Gate issuer with
the same audience.
