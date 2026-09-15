
"""SettingsAudit persistence — settings change history (append-only, for compliance auditing)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .db import SettingsAudit, Session
from .baseDB import BaseDB
from src.log import get_db_logger

log = get_db_logger()


class SettingsAuditDB(BaseDB):
    @classmethod
    def get_orm_class(cls) -> type:
        return SettingsAudit

    @classmethod
    def get_log(cls):
        return log

    @classmethod
    def ensure_table(cls) -> None:
        try:
            from db.db import get_engine
            cls.get_orm_class().__table__.create(get_engine(), checkfirst=True)
        except Exception as e:
            log.warning(f"SettingsAudit ensure_table failed: {e}")

    @classmethod
    def record(cls, key: str, action: str, old_value: Any, new_value: Any,
               changed_by: Optional[str]) -> None:
        """Write one change record (best-effort — audit failure does not block the setting change itself)."""
        try:
            with Session() as s:
                s.add(SettingsAudit(
                    key=key, action=action,
                    old_value={"v": old_value}, new_value={"v": new_value},
                    changed_by=changed_by))
                s.commit()
        except Exception as e:
            log.warning(f"SettingsAudit.record failed for {key}: {e}")

    @classmethod
    def recent(cls, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with Session() as s:
                rows = s.query(SettingsAudit).order_by(
                    SettingsAudit.changed_at.desc()).limit(limit).all()
                return [{
                    "key": r.key, "action": r.action,
                    "old": (r.old_value or {}).get("v"),
                    "new": (r.new_value or {}).get("v"),
                    "by": r.changed_by,
                    "at": r.changed_at.isoformat() if r.changed_at else None,
                } for r in rows]
        except Exception as e:
            log.warning(f"SettingsAudit.recent failed: {e}")
            return []
