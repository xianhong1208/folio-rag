
"""Tenant-isolation gate for MCP three-mode queries (verify_folder_access).

Pins down that isolation happens at the DB query layer (FolderDB.get always
carries user_token), so token A can never see token B's folder.
"""

from unittest.mock import patch

import pytest

from src.domain.exceptions import UnauthorizedAccessError
from src.domain.rag.folder_acl import verify_folder_access

_ACL = "src.domain.rag.folder_acl"


def test_owned_folder_returned_and_query_is_token_scoped():
    """Own folder -> returned; and the DB query must carry user_token (isolation at the query layer, not a post-hoc comparison)."""
    fake_folder = object()
    with patch(f"{_ACL}.FolderDB.get", return_value=[fake_folder]) as m:
        got = verify_folder_access(token="tokA", folder_name="f1")
    assert got is fake_folder
    m.assert_called_once_with(name="f1", user_token="tokA")


def test_cross_tenant_denied():
    """Another tenant's folder: querying with the caller's token returns empty -> denied (does not leak whether the folder exists)."""
    with patch(f"{_ACL}.FolderDB.get", return_value=[]):
        with pytest.raises(UnauthorizedAccessError):
            verify_folder_access(token="tokA", folder_name="tokB_folder")


def test_none_result_denied_fail_closed():
    """DB returns None (error/ownerless) is also always denied -- fail-closed."""
    with patch(f"{_ACL}.FolderDB.get", return_value=None):
        with pytest.raises(UnauthorizedAccessError):
            verify_folder_access(token="tokA", folder_name="f")
