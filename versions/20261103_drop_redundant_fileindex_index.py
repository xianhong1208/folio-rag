"""Drop idx_fileindices_file_id, a duplicate of the UNIQUE(file_id) index

FileIndices.file_id is UNIQUE, so Postgres already maintains a unique b-tree on it
("FileIndices_file_id_key"). The plain idx_fileindices_file_id from the initial
migration indexes the same column again: every insert/update maintains both and
lookups gain nothing.

The upgrade drops it only when a single-column unique index on file_id exists, so
a database that somehow lacks the constraint keeps its only index (with a NOTICE).
Idempotent in both directions (IF [NOT] EXISTS); the downgrade recreates the index
with the initial migration's name and definition, which that migration's own
downgrade expects to find.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261103d0b001'
down_revision: Union[str, None] = '20261102c0de02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


UPGRADE_SQL = r"""
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_index i
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
    WHERE i.indrelid = to_regclass('public."FileIndices"')
      AND i.indisunique AND i.indnatts = 1 AND a.attname = 'file_id'
  ) THEN
    DROP INDEX IF EXISTS public.idx_fileindices_file_id;
  ELSE
    RAISE NOTICE 'FileIndices has no unique index on file_id; keeping idx_fileindices_file_id';
  END IF;
END $$;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute('CREATE INDEX IF NOT EXISTS idx_fileindices_file_id ON "FileIndices" (file_id)')
