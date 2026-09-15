"""Folder database CRUD operations."""

from .db import Folder
from .baseDB import BaseDB
from src.log import get_db_logger

log = get_db_logger()


class FolderDB(BaseDB):
    """CRUD operations for folders."""

    @classmethod
    def get_orm_class(cls) -> type:
        return Folder

    @classmethod
    def get_log(cls):
        return log