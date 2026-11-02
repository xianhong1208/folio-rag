"""Lookup indexes on existing vector tables (node_id; file_id + chunk_index)

Neighbour expansion, read mode, chunk listing and per-file deletes filter vector
tables by ``node_id`` or by ``(metadata_->>'file_id', (metadata_->>'chunk_index')::int)``;
with only the HNSW index each was a sequential scan, which dominates query latency
once a folder holds tens of thousands of chunks. Tables created after this revision
get the same indexes from ``db_bootstrap.ensure_hnsw_for_table`` after their first
write.

Vector tables are created at runtime per folder, so this loops over whatever exists.
Each index is created in its own exception block: a table with unexpected data (e.g.
a non-integer chunk_index) skips that index with a NOTICE instead of failing the
migration and blocking startup. All statements are IF [NOT] EXISTS.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261102a1b2c4'
down_revision: Union[str, None] = '20261102a1b2c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Kept in step with src/utils/db_bootstrap.py LOOKUP_INDEX_DDL (suffixes and expressions).
UPGRADE_LOOKUP_SQL = r"""
DO $$
DECLARE
  t TEXT;
BEGIN
  FOR t IN
    SELECT c.table_name FROM information_schema.columns c
    WHERE c.table_schema = 'public' AND c.table_name LIKE 'data\_%' ESCAPE '\'
      AND c.column_name = 'node_id'
      AND EXISTS (SELECT 1 FROM information_schema.columns m
                  WHERE m.table_schema = 'public' AND m.table_name = c.table_name
                    AND m.column_name = 'metadata_')
  LOOP
    BEGIN
      EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (node_id)', t || '_nid_idx', t);
    EXCEPTION WHEN OTHERS THEN
      RAISE NOTICE 'node_id lookup index skipped for %: %', t, SQLERRM;
    END;
    BEGIN
      EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I ((metadata_->>''file_id''), ((metadata_->>''chunk_index'')::int))',
        t || '_fid_idx', t);
    EXCEPTION WHEN OTHERS THEN
      RAISE NOTICE 'file/chunk lookup index skipped for %: %', t, SQLERRM;
    END;
  END LOOP;
END $$;
"""

DOWNGRADE_LOOKUP_SQL = r"""
DO $$
DECLARE
  i TEXT;
BEGIN
  FOR i IN
    SELECT indexname FROM pg_indexes
    WHERE schemaname = 'public' AND tablename LIKE 'data\_%' ESCAPE '\'
      AND (indexname LIKE '%\_nid\_idx' ESCAPE '\' OR indexname LIKE '%\_fid\_idx' ESCAPE '\')
  LOOP
    EXECUTE format('DROP INDEX IF EXISTS %I', i);
  END LOOP;
END $$;
"""


def upgrade() -> None:
    op.execute(UPGRADE_LOOKUP_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_LOOKUP_SQL)
