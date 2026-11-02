"""FileIndices.pipeline_version: which chunking pipeline produced a file's chunks

Chunk boundaries, the text chunks are retrieved by and the contextual-prefix
prompt changed (section-aligned chunks, breadcrumbs, section-scoped prefixes).
Old indexes keep working, but only a re-index gives a file the new chunks. The
column lets the console and the REST status endpoints flag files indexed by an
older pipeline. Existing rows stay NULL, which reads as "before versions were
recorded" (re-index recommended).

Idempotent (IF NOT EXISTS), like the other migrations.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261102c0de02'
down_revision: Union[str, None] = '20261102c0de01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('ALTER TABLE "FileIndices" ADD COLUMN IF NOT EXISTS pipeline_version VARCHAR(16)')


def downgrade() -> None:
    op.execute('ALTER TABLE "FileIndices" DROP COLUMN IF EXISTS pipeline_version')
