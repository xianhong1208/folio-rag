"""QueryLogs.top_score for knowledge-gap analytics; adopt QueryLogs into alembic

top_score holds the reranker's best score for the query (before the score
threshold), so analytics can count a question as a knowledge gap when its best
match is weak, not only when it returned nothing.

QueryLogs used to be created only at runtime (QueryLogDB.ensure_table). This
migration creates it when missing (matching the QueryLog ORM in db/db.py) and
adds the column to an existing table. Idempotent: safe on a DB the runtime
already created.

"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261102a1b2c3'
down_revision: Union[str, None] = '20260819cafe01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('''
        CREATE TABLE IF NOT EXISTS "QueryLogs" (
            id SERIAL PRIMARY KEY,
            folder_id INTEGER,
            query TEXT NOT NULL,
            result_count INTEGER DEFAULT 0,
            latency_ms INTEGER,
            token_prefix VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            top_score DOUBLE PRECISION
        )
    ''')
    op.execute('ALTER TABLE "QueryLogs" ADD COLUMN IF NOT EXISTS top_score DOUBLE PRECISION')
    op.execute('CREATE INDEX IF NOT EXISTS idx_querylogs_folder ON "QueryLogs" (folder_id)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_querylogs_created ON "QueryLogs" (created_at)')


def downgrade() -> None:
    op.execute('ALTER TABLE "QueryLogs" DROP COLUMN IF EXISTS top_score')
