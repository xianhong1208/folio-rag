"""Files.metadata (flat key/value JSONB) and GIN indexes on Files.tags / Files.metadata

Folio 1.6 lets a document carry tags (already stored in ``Files.tags``) and a flat
``metadata`` object (``{"client": "Acme", "year": 2024, "signed": "2024-03-01"}``), and
filters searches and the document list by them (src/domain/rag/doc_filter.py).

Existing tag values are left exactly as they are: tags are normalised when they are
written (src/domain/rag/doc_meta.py) and matched case-insensitively, and the filter
treats a ``tags`` value that is not an array (JSON ``null``, which uploads without tags
used to store) as no tags. Existing rows get ``metadata = '{}'``.

The GIN indexes use ``jsonb_path_ops`` (smaller, and the only operator the filter uses
on them is containment ``@>``).

Idempotent (IF [NOT] EXISTS), like the other migrations.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261119f0b001'
down_revision: Union[str, None] = '20261113e0b001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        'ALTER TABLE "Files" ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT \'{}\'::jsonb')
    op.execute('CREATE INDEX IF NOT EXISTS idx_files_tags_gin '
               'ON "Files" USING gin (tags jsonb_path_ops)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_files_metadata_gin '
               'ON "Files" USING gin (metadata jsonb_path_ops)')


def downgrade() -> None:
    op.execute('DROP INDEX IF EXISTS idx_files_metadata_gin')
    op.execute('DROP INDEX IF EXISTS idx_files_tags_gin')
    op.execute('ALTER TABLE "Files" DROP COLUMN IF EXISTS metadata')
