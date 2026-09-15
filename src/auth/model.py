
"""Token management API model definitions."""

from pydantic import BaseModel
from typing import List


class TokenCreateRequest(BaseModel):
    """Token creation request model."""
    user_name: str = "generated_user"
    scopes: List[str] = ["read", "write", "admin"]
    length: int = 24
    expires_in_days: int = None


class TokenResponse(BaseModel):
    """Token response model."""
    token: str
    user_name: str
    scopes: List[str]
    expires_at: int
    expires_at_readable: str
    message: str


class TokenInfo(BaseModel):
    """Token info model."""
    user_name: str
    scopes: List[str]
    expires_at: int
    expires_at_readable: str
    module: List[str]


class TokenListResponse(BaseModel):
    """Token list response model."""
    tokens: dict[str, TokenInfo]  # Flattened: token_string -> TokenInfo
    source: str
    total_count: int
    modules: List[str]
