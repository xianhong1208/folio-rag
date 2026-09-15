
"""Ghost-folder fix.

Root cause:
(1) create: the DB row is created first, then storage name validation raises →
    a 422 goes back to the caller but the ghost row remains.
(2) delete: storage cleanup raises ValueError on an illegal name → propagates
    out → the row lives forever.

Contract:
- create fails fast on an illegal name: raises ValueError and **FolderDB.create
  must not be called**.
- delete's storage cleanup is best-effort: a ValueError does not block
  FolderDB.delete.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.adapter.folder import FolderAdapter

_F = "src.adapter.folder"


class TestCreateFailFast:
    def test_illegal_name_never_touches_db(self):
        with patch(f"{_F}.FolderDB") as db:
            with pytest.raises(ValueError, match="illegal characters"):
                FolderAdapter.create_folder(
                    name='x"><img src=x onerror=1>', user_token="tok-a")
        db.get.assert_not_called()
        db.create.assert_not_called()

    def test_illegal_token_never_touches_db(self):
        with patch(f"{_F}.FolderDB") as db:
            with pytest.raises(ValueError):
                FolderAdapter.create_folder(name="ok-name", user_token="../etc")
        db.create.assert_not_called()


class TestDeleteGhostRow:
    async def test_storage_valueerror_does_not_block_db_delete(self):
        folder = SimpleNamespace(
            id=99, name='x"><img>', user_token="tok-a",
            vector_table_uuid="u-1")
        with patch(f"{_F}.FolderDB") as db, \
             patch(f"{_F}.FileDB") as filedb, \
             patch(f"{_F}.FileStorage") as storage, \
             patch(f"{_F}.VectorStoreManager") as vsm, \
             patch(f"{_F}.notify_folder_deleted"):
            db.get.return_value = [folder]
            filedb.get.return_value = []
            storage.delete_folder.side_effect = ValueError("illegal characters")
            ok = await FolderAdapter.delete_folder(99, user_token="tok-a")
        assert ok is True
        db.delete.assert_called_once_with(id=99)  # the row must be deleted

    async def test_normal_delete_still_cleans_storage(self):
        folder = SimpleNamespace(id=7, name="good", user_token="tok-a",
                                 vector_table_uuid="u-2")
        with patch(f"{_F}.FolderDB") as db, \
             patch(f"{_F}.FileDB") as filedb, \
             patch(f"{_F}.FileStorage") as storage, \
             patch(f"{_F}.VectorStoreManager"), \
             patch(f"{_F}.notify_folder_deleted"):
            db.get.return_value = [folder]
            filedb.get.return_value = []
            ok = await FolderAdapter.delete_folder(7, user_token="tok-a")
        assert ok is True
        storage.delete_folder.assert_called_once_with("tok-a", "good")
        db.delete.assert_called_once_with(id=7)
