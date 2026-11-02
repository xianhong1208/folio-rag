"""IndexJobs.kind: jobs are no longer only indexing runs

The console's Rebuild FTS now runs as a background job ("rebuild_fts") with the same
progress / cancel / restart handling as indexing ("index"). The column records which
kind a persisted job row is, so the console can label it and a restart only marks
files of interrupted *index* jobs. Idempotent (IF [NOT] EXISTS).
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261102a1b2c3'
down_revision: Union[str, None] = '20260819cafe01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE \"IndexJobs\" ADD COLUMN IF NOT EXISTS kind VARCHAR(32) "
        "NOT NULL DEFAULT 'index'"
    )


def downgrade() -> None:
    op.execute('ALTER TABLE "IndexJobs" DROP COLUMN IF EXISTS kind')
