"""Adopt runtime-ensured schema into alembic: content_hash columns + IndexJobs table

These two schema pieces used to be added only by runtime self-healing
(fileindexdb.ensure_content_hash_columns's ALTER, indexjobdb.ensure_table's
CREATE), outside the alembic chain — so a fresh alembic-only deployment would
lack them and drift from a runtime-patched deployment. This migration adopts
them:

1. Files.content_hash / FileIndices.content_hash (VARCHAR(64) NULL)
2. IndexJobs table + three indexes (aligned with the IndexJob ORM in db/db.py)

All idempotent IF NOT EXISTS SQL: safe to re-run against a live DB the runtime
already patched (no-op). The runtime ensure_* remain as a fallback, but the
canonical definition is now here.

"""
from typing import Sequence, Union

from alembic import op


revision: str = '20260819cafe01'
down_revision: Union[str, None] = '20250127abcdef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """content_hash columns + IndexJobs table (idempotent; a no-op re-run on a runtime-patched DB)."""

    # 1. content_hash — aligned with db.py File.content_hash / FileIndex.content_hash
    op.execute('ALTER TABLE "Files" ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)')
    op.execute('ALTER TABLE "FileIndices" ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)')

    # 2. IndexJobs — aligned with the db.py IndexJob ORM (previously built by indexjobdb.ensure_table at runtime)
    #    timestamps are ISO strings (aligned with IndexJobState serialization), not a timestamp type — intentional
    op.execute('''
        CREATE TABLE IF NOT EXISTS "IndexJobs" (
            job_id UUID PRIMARY KEY,
            folder_id INTEGER NOT NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'pending',
            total_files INTEGER DEFAULT 0,
            processed_files INTEGER DEFAULT 0,
            current_index INTEGER DEFAULT 0,
            current_file_id VARCHAR(255),
            current_file_name VARCHAR(255),
            last_file_status VARCHAR(64),
            last_message TEXT,
            message TEXT,
            error TEXT,
            skip_existing BOOLEAN DEFAULT true,
            started_at VARCHAR(64),
            completed_at VARCHAR(64),
            last_updated_at VARCHAR(64),
            result_summary JSONB,
            scope_file_ids JSONB NOT NULL DEFAULT '[]',
            file_timings JSONB NOT NULL DEFAULT '[]'
        )
    ''')
    op.execute('CREATE INDEX IF NOT EXISTS idx_indexjobs_folder_id ON "IndexJobs" (folder_id)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_indexjobs_status ON "IndexJobs" (status)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_indexjobs_last_updated ON "IndexJobs" (last_updated_at)')


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS "IndexJobs"')
    op.execute('ALTER TABLE "FileIndices" DROP COLUMN IF EXISTS content_hash')
    op.execute('ALTER TABLE "Files" DROP COLUMN IF EXISTS content_hash')
