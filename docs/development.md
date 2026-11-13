# Development

[繁體中文](development.zh-TW.md)

## Architecture

- **FastMCP + FastAPI** on one ASGI app. The FastMCP app is *mounted* so its auth middleware protects `/mcp`; the FastAPI routers serve the REST API and the console.
- **PostgreSQL + pgvector** for storage; each folder gets its own `data_<folder>_<uuid>` table. Retrieval uses [LlamaIndex](https://github.com/run-llama/llama_index)'s `PGVectorStore` in hybrid mode.
- **Sync SQLAlchemy** (psycopg2) throughout — do not use `+asyncpg`.
- **Console**: Vite + React + React Router + Tailwind, built to `static/console` and served at `/admin`.

What is planned next, and why Graph RAG and "LLM Wiki" are paused: [`ROADMAP.md`](../ROADMAP.md).

## Checks

Run these before opening a pull request; CI runs the same ones.

```bash
uv sync --extra local --extra jieba --group cpu     # what CI installs
uv run --no-sync pytest --cov                       # unit tests; fails below 91% overall coverage
uv run --no-sync python scripts/check_coverage_floor.py reports/coverage.json   # every file at least 50%
uvx ruff check .                                    # correctness rules (E4, E7, E9, F)
RAG_RUN_DB_ITESTS=1 uv run --no-sync pytest tests/integration   # needs PostgreSQL + pgvector; creates and drops its own database
cd frontend && npm ci && npm test && npm run build  # console: Vitest, then the production build
```

The per-file floor reads the JSON report: add `--cov-report=json:reports/coverage.json` to the pytest
line first. The console needs Node.js 22.12+. Its build output, `static/console`, **is committed**,
and the server serves it straight from the working tree: rebuild it with every console change.

CI: *Backend CI* (`.github/workflows/backend.yml`: ruff, unit tests with coverage, the PostgreSQL
integration group, a cloud-profile install), *Console CI* (`frontend.yml`) and *Dependency audit*
(`audit.yml`: pip-audit and npm audit, also every Monday). Each runs only when its paths change.

## Releasing

1. Bump the version in `pyproject.toml` (then `uv lock`), `frontend/package.json` (then
   `npm install --package-lock-only`) and the sidebar string in `frontend/src/components/Layout.jsx`.
2. Date the `CHANGELOG.md` section, update the Shipped section of `ROADMAP.md` and
   `ROADMAP.zh-TW.md`, and rebuild the console.
3. Release pull request into `dev`; fast-forward `main` to `dev`; tag
   `git tag -a vX.Y.Z -m "Folio X.Y.Z"`; publish the GitHub Release titled `vX.Y.Z` from the
   CHANGELOG section.

`src/version.py` reads the installed package metadata first, so a running server reports the new
version only after `uv sync`.
