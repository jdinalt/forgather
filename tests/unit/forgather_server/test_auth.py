"""Tests for forgather_server.auth and the /api/auth/* routes."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect ``forgather_config_dir`` to a fresh tmp path for the test."""
    monkeypatch.setattr(
        "forgather_server.paths.forgather_config_dir", lambda: str(tmp_path)
    )
    # Reset module-level state so prior tests don't leak in.
    from forgather_server import auth

    auth._reset_sessions_for_tests()
    # Ensure auth isn't disabled by a prior test.
    auth._auth_disabled = False
    yield tmp_path


@pytest.fixture
def client(isolated_home):
    from forgather_server.app import create_app

    app = create_app()
    return TestClient(app)


class TestTokenPersistence:
    def test_load_creates_token_with_0600(self, isolated_home):
        from forgather_server import auth, paths

        token = auth.load_token()
        assert len(token) == 64
        # Hex characters only.
        assert all(c in "0123456789abcdef" for c in token)
        mode = paths.auth_token_file().stat().st_mode & 0o777
        assert mode == 0o600

    def test_load_is_idempotent(self, isolated_home):
        from forgather_server import auth

        a = auth.load_token()
        b = auth.load_token()
        assert a == b

    def test_regenerate_invalidates_old(self, isolated_home):
        from forgather_server import auth

        old = auth.load_token()
        new = auth.regenerate_token()
        assert old != new
        assert auth.verify_token(new)
        assert not auth.verify_token(old)

    def test_verify_rejects_empty(self, isolated_home):
        from forgather_server import auth

        auth.load_token()
        assert not auth.verify_token("")
        assert not auth.verify_token(None)
        assert not auth.verify_token("   ")


class TestPassword:
    def test_no_password_initially(self, isolated_home):
        from forgather_server import auth

        assert not auth.has_password()

    def test_set_then_verify(self, isolated_home):
        from forgather_server import auth, paths

        auth.set_password("hunter2")
        assert auth.has_password()
        assert auth.verify_password("hunter2")
        assert not auth.verify_password("wrong")
        mode = paths.password_hash_file().stat().st_mode & 0o777
        assert mode == 0o600

    def test_overwrites_previous(self, isolated_home):
        from forgather_server import auth

        auth.set_password("first")
        auth.set_password("second")
        assert not auth.verify_password("first")
        assert auth.verify_password("second")

    def test_empty_password_rejected(self, isolated_home):
        from forgather_server import auth

        with pytest.raises(ValueError):
            auth.set_password("")

    def test_clear_password(self, isolated_home):
        from forgather_server import auth

        auth.set_password("foo")
        auth.clear_password()
        assert not auth.has_password()
        assert not auth.verify_password("foo")

    def test_malformed_hash_rejected(self, isolated_home):
        from forgather_server import auth, paths

        paths.password_hash_file().write_text("garbage\n")
        assert not auth.verify_password("anything")

    def test_unknown_algo_rejected(self, isolated_home):
        from forgather_server import auth, paths

        paths.password_hash_file().write_text("md5$1$00$00\n")
        assert not auth.verify_password("anything")


class TestSessions:
    def test_create_and_validate(self, isolated_home):
        from forgather_server import auth

        sid = auth.create_session()
        assert auth.session_valid(sid)

    def test_revoke_invalidates(self, isolated_home):
        from forgather_server import auth

        sid = auth.create_session()
        auth.revoke_session(sid)
        assert not auth.session_valid(sid)

    def test_unknown_session_rejected(self, isolated_home):
        from forgather_server import auth

        assert not auth.session_valid("not-a-real-sid")
        assert not auth.session_valid(None)
        assert not auth.session_valid("")


class TestAuthenticateDispatch:
    def test_bearer_header_accepted(self, isolated_home):
        from forgather_server import auth

        token = auth.load_token()
        assert auth.authenticate({"authorization": f"Bearer {token}"}, {}, {})

    def test_bearer_case_insensitive_header_name(self, isolated_home):
        from forgather_server import auth

        token = auth.load_token()
        assert auth.authenticate({"Authorization": f"Bearer {token}"}, {}, {})

    def test_query_token_accepted(self, isolated_home):
        from forgather_server import auth

        token = auth.load_token()
        assert auth.authenticate({}, {"token": token}, {})

    def test_session_cookie_accepted(self, isolated_home):
        from forgather_server import auth

        sid = auth.create_session()
        assert auth.authenticate({}, {}, {auth.SESSION_COOKIE_NAME: sid})

    def test_no_credentials_rejected(self, isolated_home):
        from forgather_server import auth

        auth.load_token()
        assert not auth.authenticate({}, {}, {})

    def test_disabled_accepts_anything(self, isolated_home):
        from forgather_server import auth

        auth.load_token()
        try:
            auth.disable_auth()
            assert auth.authenticate({}, {}, {})
        finally:
            auth._auth_disabled = False


class TestPathGating:
    def test_open_paths(self, isolated_home):
        from forgather_server import auth

        for p in ("/api/health", "/api/auth/status", "/api/auth/login"):
            assert not auth.path_requires_auth(p)

    def test_static_paths_open(self, isolated_home):
        from forgather_server import auth

        for p in ("/", "/index.html", "/assets/x.js"):
            assert not auth.path_requires_auth(p)

    def test_protected_paths(self, isolated_home):
        from forgather_server import auth

        for p in (
            "/api/queue",
            "/api/jobs",
            "/api/server-identity",
            "/api/auth/set-password",
            "/api/auth/logout",
        ):
            assert auth.path_requires_auth(p)


class TestRoutes:
    def test_health_open(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_status_open(self, client):
        r = client.get("/api/auth/status")
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is False
        assert body["has_password"] is False

    def test_protected_route_returns_401_without_auth(self, client):
        r = client.get("/api/queue")
        assert r.status_code == 401
        assert r.headers.get("www-authenticate", "").startswith("Bearer")

    def test_bearer_auth_lets_through(self, client):
        from forgather_server import auth

        token = auth.load_token()
        r = client.get("/api/queue", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_query_token_lets_through(self, client):
        from forgather_server import auth

        token = auth.load_token()
        r = client.get(f"/api/queue?token={token}")
        assert r.status_code == 200

    def test_login_with_token_sets_cookie(self, client):
        from forgather_server import auth

        token = auth.load_token()
        r = client.post("/api/auth/login", json={"token": token})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        # No password set yet, so the post-login flow asks the user to set one.
        assert body["requires_password_setup"] is True
        assert auth.SESSION_COOKIE_NAME in r.cookies
        # And subsequent requests on the same client succeed without bearer.
        r = client.get("/api/queue")
        assert r.status_code == 200

    def test_login_with_wrong_token_rejected(self, client):
        r = client.post("/api/auth/login", json={"token": "wrong"})
        assert r.status_code == 401

    def test_login_with_password(self, client):
        from forgather_server import auth

        token = auth.load_token()
        # Authenticate first to set the password.
        client.post("/api/auth/login", json={"token": token})
        client.post("/api/auth/set-password", json={"password": "hunter2"})
        client.post("/api/auth/logout")
        # Cookie cleared; password should now work.
        r = client.post("/api/auth/login", json={"password": "hunter2"})
        assert r.status_code == 200
        assert r.json()["requires_password_setup"] is False

    def test_set_password_requires_session(self, client):
        # No login yet — middleware rejects before the route runs.
        r = client.post("/api/auth/set-password", json={"password": "hunter2"})
        assert r.status_code == 401

    def test_set_password_too_short_rejected(self, client):
        from forgather_server import auth

        token = auth.load_token()
        client.post("/api/auth/login", json={"token": token})
        r = client.post("/api/auth/set-password", json={"password": "ab"})
        assert r.status_code == 400

    def test_set_password_bootstrap_no_current_required(self, client):
        # Bootstrap: no password set yet, cookie session from token login
        # is enough — current_password is not required.
        from forgather_server import auth

        token = auth.load_token()
        client.post("/api/auth/login", json={"token": token})
        assert not auth.has_password()
        r = client.post("/api/auth/set-password", json={"password": "hunter2"})
        assert r.status_code == 200
        assert auth.verify_password("hunter2")

    def test_set_password_via_cookie_without_current_rejected(self, client):
        # With password set, a cookie-only session may not rotate the
        # password without re-proving knowledge of the current one.
        from forgather_server import auth

        auth.set_password("original")
        # Log in via password to obtain a session cookie.
        r = client.post("/api/auth/login", json={"password": "original"})
        assert r.status_code == 200
        r = client.post("/api/auth/set-password", json={"password": "newpass"})
        assert r.status_code == 401
        assert "current password" in r.json()["detail"]
        # Old password still works.
        assert auth.verify_password("original")

    def test_set_password_via_cookie_with_correct_current(self, client):
        from forgather_server import auth

        auth.set_password("original")
        client.post("/api/auth/login", json={"password": "original"})
        r = client.post(
            "/api/auth/set-password",
            json={"password": "newpass", "current_password": "original"},
        )
        assert r.status_code == 200
        assert auth.verify_password("newpass")
        assert not auth.verify_password("original")

    def test_set_password_via_cookie_with_wrong_current(self, client):
        from forgather_server import auth

        auth.set_password("original")
        client.post("/api/auth/login", json={"password": "original"})
        r = client.post(
            "/api/auth/set-password",
            json={"password": "newpass", "current_password": "wrong"},
        )
        assert r.status_code == 401
        assert "current password" in r.json()["detail"]
        assert auth.verify_password("original")

    def test_set_password_via_bearer_token_without_current(self, client):
        # Bearer-token auth proves possession of the on-disk token, which
        # we treat as sufficient to rotate the password.
        from forgather_server import auth

        auth.set_password("original")
        token = auth.load_token()
        client.cookies.clear()
        r = client.post(
            "/api/auth/set-password",
            json={"password": "newpass"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert auth.verify_password("newpass")

    def test_logout_clears_session(self, client):
        from forgather_server import auth

        token = auth.load_token()
        client.post("/api/auth/login", json={"token": token})
        assert client.get("/api/queue").status_code == 200
        client.post("/api/auth/logout")
        client.cookies.clear()
        r = client.get("/api/queue")
        assert r.status_code == 401

    def test_no_auth_mode_lets_everything_through(self, isolated_home):
        from forgather_server import auth
        from forgather_server.app import create_app

        try:
            auth.disable_auth()
            client = TestClient(create_app())
            assert client.get("/api/queue").status_code == 200
            # Login still works (and issues a cookie) for webui ergonomics.
            r = client.post("/api/auth/login", json={})
            assert r.status_code == 200
        finally:
            auth._auth_disabled = False


class TestCliClientToken:
    def test_loads_from_env(self, monkeypatch):
        from forgather.cli.server_client import _load_auth_token

        monkeypatch.setenv("FORGATHER_SERVER_TOKEN", "from-env")
        assert _load_auth_token() == "from-env"

    def test_loads_from_file(self, tmp_path, monkeypatch):
        from forgather.cli.server_client import _load_auth_token

        monkeypatch.delenv("FORGATHER_SERVER_TOKEN", raising=False)
        monkeypatch.setattr(
            "forgather.cli.server_client.forgather_config_dir",
            lambda: str(tmp_path),
        )
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "auth_token").write_text("from-file\n")
        assert _load_auth_token() == "from-file"

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        from forgather.cli.server_client import _load_auth_token

        monkeypatch.delenv("FORGATHER_SERVER_TOKEN", raising=False)
        monkeypatch.setattr(
            "forgather.cli.server_client.forgather_config_dir",
            lambda: str(tmp_path),
        )
        assert _load_auth_token() is None

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        from forgather.cli.server_client import _load_auth_token

        monkeypatch.setenv("FORGATHER_SERVER_TOKEN", "winner")
        monkeypatch.setattr(
            "forgather.cli.server_client.forgather_config_dir",
            lambda: str(tmp_path),
        )
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "auth_token").write_text("loser\n")
        assert _load_auth_token() == "winner"
