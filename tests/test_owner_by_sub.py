"""Folders belong to the person (the JWT ``sub``), not to one token (its ``jti``).

Covers the owner key (src/auth/owner_key.py), the storage directory it maps to, and
that every entry point — REST, the console and ``/mcp`` (tool list and tool calls) —
derives the same ``sub:<sub>`` key from any token of the same person, refuses a token
without a ``sub``, and claims legacy folders only for a verified token.

Token shapes follow what MCP Gate issues (MCP_Gate src/oauth/service.py):
- personal access token: sub = the user's id (a UUID), client_id = mcp-gate-console
- OAuth (authorization code / refresh): sub = the same user id, client_id = the client
- client_credentials: sub = the client's own client_id (mcpc_…)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth.owner_key import (
    MissingSubjectError,
    OWNER_PREFIX,
    TokenOwner,
    is_person_key,
    owner_from_bearer,
    owner_key_for_sub,
    owner_key_from_bearer,
)
from src.domain.exceptions import InvalidTokenError

USER = "5d1c9f0e-7a3b-4c1d-9e8f-0123456789ab"   # MCP Gate admin_users.id
OTHER_USER = "a0b1c2d3-e4f5-4a6b-8c7d-0123456789ff"
CLIENT = "mcpc_0123456789abcdef01234567"          # generate_client_id()


def _pat(tokens, sub=USER, **kw):
    return tokens.make(sub=sub, client_id="mcp-gate-console", **kw)


def _oauth(tokens, sub=USER, **kw):
    return tokens.make(sub=sub, client_id=CLIENT, **kw)


# -- owner key ---------------------------------------------------------------------------


class TestOwnerKey:
    def test_pat_and_oauth_token_of_one_user_share_the_key(self, tokens):
        pat, oauth = _pat(tokens), _oauth(tokens)
        assert owner_key_from_bearer(pat) == owner_key_from_bearer(oauth) == f"sub:{USER}"

    def test_each_token_keeps_its_own_jti(self, tokens):
        a = owner_from_bearer(_pat(tokens, jti="j-1"))
        b = owner_from_bearer(_oauth(tokens, jti="j-2"))
        assert (a.jti, b.jti) == ("j-1", "j-2") and a.owner_key == b.owner_key

    def test_rotated_oauth_token_keeps_the_key(self, tokens):
        keys = {owner_key_from_bearer(_oauth(tokens)) for _ in range(3)}  # new jti each time
        assert keys == {f"sub:{USER}"}

    def test_another_user_gets_another_key(self, tokens):
        assert owner_key_from_bearer(_pat(tokens, sub=OTHER_USER)) != f"sub:{USER}"

    def test_client_credentials_token_is_its_client(self, tokens):
        assert owner_key_from_bearer(tokens.make(sub=CLIENT, client_id=CLIENT)) == f"sub:{CLIENT}"

    @pytest.mark.parametrize("claims", [{"jti": "j"}, {"jti": "j", "sub": ""},
                                        {"jti": "j", "sub": "  "}, {"jti": "j", "sub": None}])
    def test_token_without_sub_is_refused(self, claims):
        tok = jwt.encode(claims, "k" * 32, algorithm="HS256")
        with pytest.raises(MissingSubjectError) as e:
            owner_key_from_bearer(tok)
        assert isinstance(e.value, InvalidTokenError)
        assert "no subject" in str(e.value)

    def test_opaque_token_is_refused(self):
        with pytest.raises(MissingSubjectError):
            owner_key_from_bearer("opaque-token")

    def test_prefix_keeps_person_and_legacy_keys_apart(self):
        assert OWNER_PREFIX == "sub:"
        key = owner_key_for_sub("x")
        assert key == "sub:x" and is_person_key(key)
        assert not is_person_key("8f14e45f-ceea-4f6a-9c43-1f2c3d4e5f60")  # a legacy jti
        assert not is_person_key(None) and not is_person_key("console")

    def test_a_sub_that_looks_like_a_jti_still_gets_the_prefix(self):
        jti_like = "8f14e45f-ceea-4f6a-9c43-1f2c3d4e5f60"
        assert owner_key_for_sub(jti_like) != jti_like

    def test_long_sub_fits_the_column(self):
        key = owner_key_for_sub("u" * 400)
        assert key.startswith("sub:sha256:") and len(key) <= 256
        assert owner_key_for_sub("u" * 252) == "sub:" + "u" * 252
        assert owner_key_for_sub("u" * 400) != owner_key_for_sub("u" * 401)


# -- storage directory -------------------------------------------------------------------


class TestOwnerDir:
    def test_person_key_maps_to_a_safe_directory(self):
        from src.storage.file_storage import owner_dir_name
        assert owner_dir_name(f"sub:{USER}") == f"sub_{USER}"

    def test_unsafe_sub_is_hashed(self):
        from src.storage.file_storage import owner_dir_name
        d = owner_dir_name("sub:https://client.example/a")
        assert d.startswith("subh_") and "/" not in d
        assert owner_dir_name("sub:" + "u" * 201).startswith("subh_")

    def test_legacy_key_keeps_its_directory(self):
        from src.storage.file_storage import owner_dir_name
        assert owner_dir_name("8f14e45f-ceea") == "8f14e45f-ceea"
        assert owner_dir_name("console") == "console"
        with pytest.raises(ValueError):
            owner_dir_name("../etc")

    def test_files_are_stored_under_the_person(self, tmp_path, monkeypatch):
        from src.storage.file_storage import FileStorage
        monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path / "storage")
        monkeypatch.setattr(FileStorage, "STORAGE_FILE_ROOT", tmp_path)
        rel = FileStorage.save_file(f"sub:{USER}", "docs", "f1", b"x")
        assert rel == f"storage/sub_{USER}/docs/f1"
        assert (tmp_path / rel).read_bytes() == b"x"

    def test_move_between_owners(self, tmp_path, monkeypatch):
        from src.storage.file_storage import FileStorage
        monkeypatch.setattr(FileStorage, "STORAGE_ROOT", tmp_path)
        (tmp_path / "jti-1" / "docs").mkdir(parents=True)
        (tmp_path / "jti-1" / "docs" / "f").write_text("x")
        assert FileStorage.move_folder("jti-1", "docs", "sub:u", "docs-2")
        assert (tmp_path / "sub_u" / "docs-2" / "f").read_text() == "x"
        assert not (tmp_path / "jti-1" / "docs").exists()
        # target taken -> refused, nothing moved
        (tmp_path / "jti-2" / "docs").mkdir(parents=True)
        assert not FileStorage.move_folder("jti-2", "docs", "sub:u", "docs-2")
        assert (tmp_path / "jti-2" / "docs").is_dir()


# -- REST --------------------------------------------------------------------------------


@pytest.fixture()
def rest(gate_verifier):
    """The real folder router behind the real auth dependency (stub JWKS)."""
    from src.api.router import folder_api
    from src.auth.dependencies import set_token_verifier
    from src.middleware.error_handler import register_error_handlers
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(folder_api.router)
    set_token_verifier(gate_verifier)
    with patch.object(folder_api.FolderAdapter, "get_folder", new=AsyncMock(return_value=[])) as gf, \
         patch("src.adapter.folder_claim.claim_for", new=AsyncMock(return_value=[])) as claim:
        yield SimpleNamespace(client=TestClient(app), get_folder=gf, claim=claim)
    set_token_verifier(None)


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


class TestRest:
    def test_two_tokens_of_one_user_list_the_same_folders(self, rest, tokens):
        for tok in (_pat(tokens), _oauth(tokens)):
            assert rest.client.get("/folders/", headers=_bearer(tok)).status_code == 200
        owners = [c.kwargs["user_token"] for c in rest.get_folder.await_args_list]
        assert owners == [f"sub:{USER}", f"sub:{USER}"]

    def test_another_user_is_scoped_to_their_own(self, rest, tokens):
        rest.client.get("/folders/", headers=_bearer(_pat(tokens, sub=OTHER_USER)))
        assert rest.get_folder.await_args.kwargs["user_token"] == f"sub:{OTHER_USER}"

    def test_token_without_sub_is_401(self, rest, tokens):
        r = rest.client.get("/folders/", headers=_bearer(tokens.make(drop=("sub",))))
        assert r.status_code == 401 and "no subject" in r.text
        assert not rest.get_folder.await_count and not rest.claim.await_count

    def test_verified_token_claims_its_legacy_folders(self, rest, tokens):
        rest.client.get("/folders/", headers=_bearer(_pat(tokens, jti="j-legacy")))
        (owner,), _ = rest.claim.await_args
        assert owner == TokenOwner(owner_key=f"sub:{USER}", sub=USER, jti="j-legacy")

    def test_unverified_token_never_claims(self, rest, tokens):
        forged = _pat(tokens, key=tokens.other_key)  # signed by someone else
        assert rest.client.get("/folders/", headers=_bearer(forged)).status_code == 401
        assert not rest.claim.await_count

    async def test_extract_token_claims_only_the_token_verification_marked(self, tokens):
        from src.api.dependencies.auth import extract_token
        tok = _pat(tokens)
        req = SimpleNamespace(headers={"Authorization": f"Bearer {tok}"},
                              state=SimpleNamespace(verified_bearer="another-token"))
        with patch("src.adapter.folder_claim.claim_for", new=AsyncMock()) as claim:
            assert await extract_token(req) == f"sub:{USER}"
            assert not claim.await_count
            req.state.verified_bearer = tok
            assert await extract_token(req) == f"sub:{USER}"
            assert claim.await_count == 1


# -- /mcp: tool list ---------------------------------------------------------------------


@pytest.fixture()
def mcp_filter(gate_verifier):
    """An AuthFilteredFastMCP with Folio_docs registered for USER and Folio_notes for
    OTHER_USER; ``run(token)`` returns the tool names that token is shown."""
    from src.auth.dependencies import set_token_verifier
    from src.fastmcp_tools import auth_filtered_mcp as afm
    from src.fastmcp_tools.dynamic_tool_manager import DynamicToolManager

    folders = {f"sub:{USER}": [SimpleNamespace(name="docs")],
               f"sub:{OTHER_USER}": [SimpleNamespace(name="notes")]}
    DynamicToolManager._instance = None
    m = DynamicToolManager.initialize(MagicMock())
    m._registered_tools, m._token_folder_to_tool = {}, {}
    m._token_folder_descriptions, m._tool_to_folder = {}, {}
    cfg = SimpleNamespace(app=SimpleNamespace(tool_prefix="Folio"))
    set_token_verifier(gate_verifier)
    afm._filter_cache.clear()
    get = lambda user_token=None, **kw: folders.get(user_token, [])  # noqa: E731
    with patch("src.fastmcp_tools.dynamic_tool_manager.Config.get_config_model", return_value=cfg), \
         patch("src.fastmcp_tools.dynamic_tool_manager.FolderDB.get", side_effect=get), \
         patch.object(afm.FolderDB, "get", side_effect=get), \
         patch("src.adapter.folder_claim.claim_for", new=AsyncMock(return_value=[])) as claim:
        m.register_folder_tool("docs", MagicMock(), user_token=f"sub:{USER}")
        m.register_folder_tool("notes", MagicMock(), user_token=f"sub:{OTHER_USER}")
        srv = afm.AuthFilteredFastMCP("t")
        all_tools = [SimpleNamespace(name=n, description="d", model_copy=lambda update, n=n:
                                     SimpleNamespace(name=n, **update))
                     for n in ("Folio_docs", "Folio_notes", "health")]

        async def run(token):
            req = SimpleNamespace(headers={"Authorization": f"Bearer {token}"})
            with patch.object(afm, "get_http_request", return_value=req):
                return sorted(t.name for t in await srv._apply_token_filter(list(all_tools)))
        yield SimpleNamespace(run=run, claim=claim)
    set_token_verifier(None)
    afm._filter_cache.clear()
    DynamicToolManager._instance = None


class TestMcpToolList:
    async def test_two_tokens_of_one_user_see_the_same_tools(self, mcp_filter, tokens):
        assert await mcp_filter.run(_pat(tokens)) == ["Folio_docs", "health"]
        assert await mcp_filter.run(_oauth(tokens)) == ["Folio_docs", "health"]

    async def test_another_user_sees_only_theirs(self, mcp_filter, tokens):
        assert await mcp_filter.run(_oauth(tokens, sub=OTHER_USER)) == ["Folio_notes", "health"]
        assert await mcp_filter.run(_oauth(tokens, sub="nobody")) == ["health"]

    async def test_token_without_sub_sees_no_folder_tools(self, mcp_filter, tokens):
        assert await mcp_filter.run(tokens.make(drop=("sub",))) == ["health"]
        assert not mcp_filter.claim.await_count

    async def test_list_claims_for_the_verified_token(self, mcp_filter, tokens):
        await mcp_filter.run(_oauth(tokens, jti="j-1"))
        (owner,), _ = mcp_filter.claim.await_args
        assert (owner.owner_key, owner.jti) == (f"sub:{USER}", "j-1")

    async def test_invalid_token_never_claims(self, mcp_filter, tokens):
        assert await mcp_filter.run(_oauth(tokens, key=tokens.other_key)) == ["health"]
        assert not mcp_filter.claim.await_count


# -- /mcp: tool calls --------------------------------------------------------------------


class TestMcpToolCall:
    def _req(self, token):
        return SimpleNamespace(headers={"Authorization": f"Bearer {token}"})

    async def test_verified_token_is_owned_by_sub_and_claims(self, tokens):
        from src.fastmcp_tools import acl
        tok = _oauth(tokens, jti="j-9")
        access = SimpleNamespace(token=tok, claims=jwt.decode(tok, options={"verify_signature": False}))
        with patch.object(acl, "get_http_request", return_value=self._req(tok)), \
             patch.object(acl, "get_access_token", return_value=access), \
             patch("src.adapter.folder_claim.claim_for", new=AsyncMock()) as claim:
            assert await acl.extract_owner_key() == f"sub:{USER}"
        (owner,), _ = claim.await_args
        assert owner.jti == "j-9"

    async def test_owner_comes_from_the_verified_claims(self, tokens):
        """The provider's verified claims win over whatever the raw header decodes to."""
        from src.fastmcp_tools import acl
        tok = _oauth(tokens)
        access = SimpleNamespace(token=tok, claims={"sub": OTHER_USER, "jti": "j"})
        with patch.object(acl, "get_http_request", return_value=self._req(tok)), \
             patch.object(acl, "get_access_token", return_value=access), \
             patch("src.adapter.folder_claim.claim_for", new=AsyncMock()):
            assert await acl.extract_owner_key() == f"sub:{OTHER_USER}"

    async def test_unverified_token_never_claims(self, tokens):
        from src.fastmcp_tools import acl
        tok = _oauth(tokens)
        for access in (None, SimpleNamespace(token="other", claims={"sub": "x"})):
            with patch.object(acl, "get_http_request", return_value=self._req(tok)), \
                 patch.object(acl, "get_access_token", return_value=access), \
                 patch("src.adapter.folder_claim.claim_for", new=AsyncMock()) as claim:
                assert await acl.extract_owner_key() == f"sub:{USER}"
            assert not claim.await_count

    async def test_token_without_sub_is_unauthorized(self, tokens):
        from src.fastmcp_tools import acl
        with patch.object(acl, "get_http_request",
                          return_value=self._req(tokens.make(drop=("sub",)))), \
             patch.object(acl, "get_access_token", return_value=None):
            with pytest.raises(InvalidTokenError):
                await acl.extract_owner_key()
            with pytest.raises(InvalidTokenError):
                acl.extract_token()

    async def test_tool_call_without_sub_returns_unauthorized(self, tokens):
        import json
        from src.fastmcp_tools import acl
        from src.fastmcp_tools import agentic_tools as at
        with patch.object(acl, "get_http_request",
                          return_value=self._req(tokens.make(drop=("sub",)))), \
             patch.object(acl, "get_access_token", return_value=None), \
             patch.object(at, "get_rag_adapter") as rag:
            out = await at._query(folder_name="docs", mode="list", query=None, file_id=None,
                                  top_k=None, similarity_cutoff=None, expand_context=None)
        body = json.loads(out[0].text)
        assert body["error"] == "unauthorized" and "no subject" in body["detail"]
        assert not rag.called


# -- console -----------------------------------------------------------------------------


ORIGIN = {"X-RAG-Console": "1", "Origin": "http://testserver"}


@pytest.fixture()
def console_app(console_auth):
    from src.api.router.admin_settings import router
    app = FastAPI()
    app.include_router(router)
    with patch("src.adapter.folder_claim.claim_for", new=AsyncMock(return_value=[])) as claim:
        yield SimpleNamespace(client=TestClient(app), claim=claim)


class TestConsole:
    def test_identity_owner_key_is_the_sub(self, make_identity):
        ident = make_identity("sub:alice")
        assert ident.owner_key == "sub:alice"
        assert ident.owns("sub:alice") and not ident.owns(ident.jti)
        assert not ident.owns("sub:bob") and not ident.owns(None)

    def test_open_mode_unchanged(self, open_console_auth):
        ident = open_console_auth.open_identity()
        assert ident.owner_key is None and ident.owns("anything") and ident.token_owner is None

    def test_two_pats_of_one_user_see_the_same_folders(self, console_app, tokens):
        seen = []
        with patch("db.admin_stats.folders_with_index_stats",
                   side_effect=lambda owner: seen.append(owner) or []):
            for tok in (_pat(tokens), _pat(tokens)):
                c = TestClient(console_app.client.app)
                assert c.post("/api/console/session", json={"token": tok},
                              headers=ORIGIN).status_code == 200
                assert c.get("/api/admin/folders").status_code == 200
        assert seen == [f"sub:{USER}", f"sub:{USER}"]

    def test_pat_without_sub_cannot_sign_in(self, console_app, tokens):
        r = console_app.client.post("/api/console/session",
                                    json={"token": tokens.make(drop=("sub",))}, headers=ORIGIN)
        assert r.status_code == 401 and r.json()["detail"]["error"] == "missing_sub"
        r = console_app.client.get("/api/admin/folders",
                                   headers=_bearer(tokens.make(drop=("sub",))))
        assert r.status_code == 401
        assert not console_app.claim.await_count

    def test_session_cookie_without_sub_is_refused(self, console_auth):
        import time
        from fastapi import HTTPException
        from src.auth.console_session import SESSION_TYP
        now = int(time.time())
        value = jwt.encode({"typ": SESSION_TYP, "jti": "j", "sid": "s", "iat": now,
                            "exp": now + 60, "scopes": []}, console_auth.secret, algorithm="HS256")
        with pytest.raises(HTTPException) as e:
            console_auth.read_session(value)
        assert e.value.status_code == 401

    def test_identity_without_sub_never_scopes_to_everything(self):
        from src.auth.console_session import ConsoleIdentity
        ident = ConsoleIdentity(mode="token", via="cookie", jti="j", sub=None)
        with pytest.raises(MissingSubjectError):
            ident.owner_key  # noqa: B018 - would otherwise mean "every folder"
        assert not ident.owns("sub:None") and not ident.owns("j")

    def test_sign_in_bearer_and_cookie_claim_for_the_token(self, console_app, tokens):
        tok = _pat(tokens, jti="j-console")
        c = console_app.client
        assert c.post("/api/console/session", json={"token": tok}, headers=ORIGIN).status_code == 200
        with patch("db.admin_stats.folders_with_index_stats", return_value=[]):
            c.get("/api/admin/folders")                                     # cookie
            TestClient(c.app).get("/api/admin/folders", headers=_bearer(tok))  # bearer
        owners = [a.args[0] for a in console_app.claim.await_args_list]
        assert len(owners) == 3
        assert {(o.owner_key, o.jti) for o in owners} == {(f"sub:{USER}", "j-console")}

    def test_open_mode_never_claims_and_creates_as_console(self, open_console_auth):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        with patch("src.adapter.folder_claim.claim_for", new=AsyncMock()) as claim, \
             patch("src.api.router.folder_api.create_folder",
                   new=AsyncMock(return_value={"id": 1})) as create:
            r = TestClient(app).post("/api/admin/manage/folders", json={"name": "n"})
        assert r.status_code == 200 and create.await_args.kwargs["user_token"] == "console"
        assert not claim.await_count


class TestLegacyCount:
    def _overview(self, client):
        base = {"folders": 0, "files": 0, "total_size_bytes": 0, "index": {}, "jobs": {}}
        with patch("db.admin_stats.overview", return_value=dict(base)), \
             patch("db.admin_stats.indexing_trend", return_value=[]), \
             patch("src.config.runtime_overrides.get_effective", return_value={}), \
             patch("src.config.config_manager.Config.get_config_model",
                   return_value=SimpleNamespace()), \
             patch("db.folder_claim_db.count_legacy", return_value=4) as count:
            return client.get("/api/admin/overview").json()["data"], count

    def _client(self, as_identity, ident):
        from src.api.router.admin_settings import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(as_identity(app, ident))

    def test_admin_sees_how_many_legacy_folders_remain(self, as_identity, make_identity):
        data, count = self._overview(self._client(as_identity, make_identity(admin=True)))
        assert data["legacy_owned_folders"] == 4
        count.assert_called_once_with("console")

    def test_non_admin_does_not(self, as_identity, make_identity):
        data, count = self._overview(self._client(as_identity, make_identity(admin=False)))
        assert "legacy_owned_folders" not in data and not count.called

    def test_count_failure_is_reported_as_unknown(self):
        from src.adapter.folder_claim import count_legacy_folders
        with patch("db.folder_claim_db.count_legacy", side_effect=RuntimeError("db down")):
            assert count_legacy_folders() is None
