"""Remove FK constraint and drop auth_tokens table for remote auth

This migration:
1. Removes the FK constraint from Folders.user_token -> auth_tokens.token
2. Drops the auth_tokens table (no longer needed with remote Token Server)
3. Adds an index on Folders.user_token for faster filtering

Revision ID: 20250127_remove_fk
Revises: 1e754db7cbd1
Create Date: 2025-01-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func


# revision identifiers, used by Alembic.
revision: str = '20250127abcdef'
down_revision: Union[str, None] = '1e754db7cbd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove FK constraint, drop auth_tokens, add index"""

    # 1. Remove the FK constraint (must be done before dropping the table)
    # PostgreSQL auto-generates FK names as: {table}_{column}_fkey
    try:
        op.drop_constraint('Folders_user_token_fkey', 'Folders', type_='foreignkey')
    except Exception:
        # Try alternative naming convention
        try:
            op.drop_constraint('fk_folders_user_token', 'Folders', type_='foreignkey')
        except Exception:
            # If no FK exists, that's fine
            pass

    # 2. Drop the auth_tokens table (no longer needed)
    op.drop_table('auth_tokens')

    # 3. Add index on user_token for faster filtering queries
    op.create_index('idx_folders_user_token', 'Folders', ['user_token'], unique=False)


def downgrade() -> None:
    """Restore auth_tokens table and FK constraint"""

    # Remove the index
    op.drop_index('idx_folders_user_token', table_name='Folders')

    # Recreate auth_tokens table
    op.create_table(
        'auth_tokens',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('module', sa.String(64), nullable=False),
        sa.Column('token', sa.String(256), nullable=False),
        sa.Column('user_name', sa.String(64), nullable=False),
        sa.Column('scopes', sa.String(256), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=func.now(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )

    # Restore FK constraint
    op.create_foreign_key(
        'Folders_user_token_fkey',
        'Folders',
        'auth_tokens',
        ['user_token'],
        ['token']
    )
