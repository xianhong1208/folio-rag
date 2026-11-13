"""FileIndices.embedding_dimension: the vector length a file was indexed with

FileIndices already records the embedding model each file was indexed with; this
adds its dimension, so a file indexed by another embedding model or dimension than
the configured one can be flagged "re-index recommended" (its vectors and the
query's are not comparable).

Existing rows are backfilled from their folder's vector table (``vector_store_table``,
with or without llama-index's ``data_`` prefix): pgvector stores the dimension of a
``vector(N)`` column as the attribute's type modifier. Rows whose
table no longer exists (or has no fixed dimension) stay NULL, which flags nothing.

Idempotent (IF NOT EXISTS; the backfill only fills NULLs), like the other migrations.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261113e0b001'
down_revision: Union[str, None] = '20261103d0b001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


BACKFILL_SQL = r"""
UPDATE "FileIndices" AS fi
SET embedding_dimension = a.atttypmod
FROM pg_attribute AS a
WHERE fi.embedding_dimension IS NULL
  AND fi.vector_store_table IS NOT NULL
  AND fi.vector_store_table <> ''
  AND a.attrelid = COALESCE(
        to_regclass(format('%I', fi.vector_store_table)),
        to_regclass(format('%I', 'data_' || fi.vector_store_table)))
  AND a.attname = 'embedding'
  AND NOT a.attisdropped
  AND a.atttypmod > 0
"""


def upgrade() -> None:
    op.execute('ALTER TABLE "FileIndices" ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER')
    op.execute(BACKFILL_SQL)


def downgrade() -> None:
    op.execute('ALTER TABLE "FileIndices" DROP COLUMN IF EXISTS embedding_dimension')
