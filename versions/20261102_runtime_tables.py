"""Adopt RuntimeSettings, SettingsAudit and QueryLogs into Alembic

These three tables were only ever created at startup by ``ensure_table()``
(``Base.metadata`` create), outside the migration chain, so later schema changes
to them had no migration path and an Alembic-only deployment lacked them. The
definitions below match the ORM in db/db.py. All statements are IF NOT EXISTS,
so this is a no-op on a database the runtime already created them in. The
runtime ensure_table() calls remain only as a fallback for ``--no-migrate``.
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20261102a1b2c5'
down_revision: Union[str, None] = '20261102a1b2c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('''
        CREATE TABLE IF NOT EXISTS "RuntimeSettings" (
            key VARCHAR(255) PRIMARY KEY,
            value JSONB NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_by VARCHAR(64)
        )
    ''')
    op.execute('''
        CREATE TABLE IF NOT EXISTS "SettingsAudit" (
            id SERIAL PRIMARY KEY,
            key VARCHAR(255) NOT NULL,
            action VARCHAR(16) NOT NULL,
            old_value JSONB,
            new_value JSONB,
            changed_by VARCHAR(64),
            changed_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        )
    ''')
    op.execute('CREATE INDEX IF NOT EXISTS idx_settings_audit_at ON "SettingsAudit" (changed_at)')
    op.execute('''
        CREATE TABLE IF NOT EXISTS "QueryLogs" (
            id SERIAL PRIMARY KEY,
            folder_id INTEGER,
            query TEXT NOT NULL,
            result_count INTEGER,
            latency_ms INTEGER,
            token_prefix VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        )
    ''')
    op.execute('CREATE INDEX IF NOT EXISTS idx_querylogs_folder ON "QueryLogs" (folder_id)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_querylogs_created ON "QueryLogs" (created_at)')


def downgrade() -> None:
    op.execute('DROP TABLE IF EXISTS "QueryLogs"')
    op.execute('DROP TABLE IF EXISTS "SettingsAudit"')
    op.execute('DROP TABLE IF EXISTS "RuntimeSettings"')
