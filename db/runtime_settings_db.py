
"""RuntimeSettings persistence — the override layer for settings hot-changed
from the admin panel.

config.yaml holds the read-only factory defaults; this table holds live
adjustments (dotted-path key -> value). At startup main.py reads them back and
overlays them onto the ConfigModel; every subsequent admin change is
write-through.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .db import RuntimeSetting
from db.db import Session as DBSession
from .baseDB import BaseDB
from src.log import get_db_logger

log = get_db_logger()


class RuntimeSettingsDB(BaseDB):
    """RuntimeSettings CRUD (key = dotted setting path, value = {"value": ...})."""

    @classmethod
    def get_orm_class(cls) -> type:
        return RuntimeSetting

    @classmethod
    def get_log(cls):
        return log

    @classmethod
    def ensure_table(cls) -> None:
        """Create the table (IF NOT EXISTS, idempotent) — a runtime safety net
        beyond the bootstrap create_all."""
        try:
            from db.db import get_engine
            cls.get_orm_class().__table__.create(get_engine(), checkfirst=True)
        except Exception as e:
            cls.get_log().warning(f"RuntimeSettings ensure_table failed: {e}")

    @classmethod
    def load_all(cls) -> Dict[str, Any]:
        """All overrides: {dotted_path: value}. If the table is missing or the DB
        is down, returns an empty dict (does not block startup)."""
        try:
            with DBSession() as session:
                rows = session.query(RuntimeSetting).all()
                return {r.key: (r.value or {}).get("value") for r in rows}
        except Exception as e:
            log.warning(f"RuntimeSettings load_all failed (using yaml defaults only): {e}")
            return {}

    @classmethod
    def upsert(cls, key: str, value: Any, updated_by: Optional[str] = None) -> None:
        with DBSession() as session:
            row = session.get(RuntimeSetting, key)
            if row is None:
                session.add(RuntimeSetting(
                    key=key, value={"value": value}, updated_by=updated_by))
            else:
                row.value = {"value": value}
                row.updated_by = updated_by
            session.commit()

    @classmethod
    def delete(cls, key: str) -> bool:
        """Remove a single override (reverting to the yaml default). Returns
        whether a row was actually deleted."""
        with DBSession() as session:
            row = session.get(RuntimeSetting, key)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True
