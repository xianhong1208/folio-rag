"""FileIndices.index_warning: a caveat on an indexed file

A file can be indexed and still be incomplete: docling reports pages it could
not parse (a stage error or an out-of-memory error swallowed into a partial
result). Those page numbers used to be dropped silently; they are now stored
here and shown in the console and the REST status endpoints.

Idempotent (IF NOT EXISTS), like the other migrations.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261102c0de01'
down_revision: Union[str, None] = '20261102a1b2c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('ALTER TABLE "FileIndices" ADD COLUMN IF NOT EXISTS index_warning TEXT')


def downgrade() -> None:
    op.execute('ALTER TABLE "FileIndices" DROP COLUMN IF EXISTS index_warning')
