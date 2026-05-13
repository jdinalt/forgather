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


class TestPeerMutualTLS:
    """Exercise the AuthMiddleware peer carve-out gate (issue #31).

    Drives the middleware directly with crafted ASGI scopes so the
    decision matrix (bearer / client-cert / peer-IP / nothing) can be
    tested without standing up a real TLS listener.
    """

    @pytest.fixture
    def driven_middleware(self, isolated_home, monkeypatch):
        """Return (call, captured) where call(scope) invokes the gate
        and captured["app_ran"] tells you whether the downstream app
        was reached."""
        from forgather_server import auth

        captured = {"app_ran": False, "send": []}

        async def inner_app(scope, receive, send):
            captured["app_ran"] = True

        async def fake_receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def fake_send(message):
            captured["send"].append(message)

        mw = auth.AuthMiddleware(inner_app)

        def call(scope):
            import asyncio

            captured["app_ran"] = False
            captured["send"] = []
            asyncio.run(mw(scope, fake_receive, fake_send))
            return captured

        return call, captured

    @staticmethod
    def _http_scope(
        path: str,
        *,
        method: str = "GET",
        client: tuple = ("10.0.0.5", 54321),
        headers=None,
        tls_ext: dict | None = None,
    ) -> dict:
        scope: dict = {
            "type": "http",
            "method": method,
            "path": path,
            "client": client,
            "headers": headers or [],
            "query_string": b"",
        }
        if tls_ext is not None:
            scope["extensions"] = {"tls": tls_ext}
        return scope

    def test_client_cert_allows_peer_path(self, driven_middleware):
        """A verified client cert authenticates an inter-node GET."""
        call, captured = driven_middleware
        scope = self._http_scope(
            "/api/cluster/members",
            tls_ext={"client_cert_verified": True},
        )
        call(scope)
        assert captured["app_ran"] is True

    def test_client_cert_allows_peer_mutation(self, driven_middleware):
        call, captured = driven_middleware
        scope = self._http_scope(
            "/api/cluster/training_local",
            method="POST",
            tls_ext={"client_cert_verified": True},
        )
        call(scope)
        assert captured["app_ran"] is True

    def test_client_cert_does_not_open_user_paths(self, driven_middleware):
        """A client cert is *not* a substitute for a user bearer.

        The cert proves "I am a cluster member"; it does NOT grant
        access to user-facing endpoints (those still need a bearer).
        """
        call, captured = driven_middleware
        scope = self._http_scope(
            "/api/queue",
            tls_ext={"client_cert_verified": True},
        )
        call(scope)
        assert captured["app_ran"] is False
        assert any(
            m["type"] == "http.response.start" and m["status"] == 401
            for m in captured["send"]
        )

    def test_no_credentials_rejected_on_peer_path(self, driven_middleware):
        """No bearer, no cert → 401 even on a peer-allowed path."""
        call, captured = driven_middleware
        scope = self._http_scope("/api/cluster/members")
        call(scope)
        assert captured["app_ran"] is False

    def test_peer_ip_alone_does_not_authenticate(self, driven_middleware):
        """Source IP is not authentication.

        Even a request from a putative peer address must present a
        verified client cert to reach an inter-node endpoint.
        """
        call, captured = driven_middleware
        scope = self._http_scope(
            "/api/cluster/members",
            client=("10.0.0.5", 12345),
        )
        call(scope)
        assert captured["app_ran"] is False

    def test_unverified_cert_does_not_authenticate(self, driven_middleware):
        """A presented-but-not-verified cert must not pass the gate.

        With ``ssl_cert_reqs=CERT_OPTIONAL`` + ``ssl_ca_certs`` we
        only ever get ``client_cert_verified=True`` on the scope, but
        a buggy protocol could in theory set it falsy. Make sure the
        check is on the truthy verified flag, not on cert presence
        alone.
        """
        call, captured = driven_middleware
        scope = self._http_scope(
            "/api/cluster/members",
            tls_ext={
                "client_cert_chain_der": [b"fake-der"],
                "client_cert_verified": False,
            },
        )
        call(scope)
        assert captured["app_ran"] is False

    def test_bearer_still_works_alongside_cert_path(self, driven_middleware):
        """A valid bearer still authenticates regardless of cert state."""
        from forgather_server import auth

        token = auth.load_token()
        call, captured = driven_middleware
        scope = self._http_scope(
            "/api/queue",
            headers=[(b"authorization", f"Bearer {token}".encode())],
        )
        call(scope)
        assert captured["app_ran"] is True


class TestTLSHelpers:
    def test_httpx_client_cert_returns_none_when_unprovisioned(self, tmp_path, monkeypatch):
        """When the local node has no cert+key on disk, return None."""
        from forgather.tls import httpx_client_cert
        from forgather.tls.config import TLSConfig

        cfg = TLSConfig(root=tmp_path)  # no files
        # Sanity: explicit cfg path. Also test the implicit-load branch
        # by pointing the env override at an empty dir.
        monkeypatch.setenv("FORGATHER_TLS_DIR", str(tmp_path))
        assert httpx_client_cert() is None

    def test_httpx_client_cert_returns_pair_when_provisioned(
        self, tmp_path, monkeypatch
    ):
        """When cert+key exist, return their paths as a tuple."""
        from forgather.tls import httpx_client_cert

        (tmp_path / "server.crt").write_text("-----BEGIN CERTIFICATE-----\n")
        (tmp_path / "server.key").write_text("-----BEGIN PRIVATE KEY-----\n")
        monkeypatch.setenv("FORGATHER_TLS_DIR", str(tmp_path))
        result = httpx_client_cert()
        assert result is not None
        cert_path, key_path = result
        assert cert_path.endswith("server.crt")
        assert key_path.endswith("server.key")

    def test_uvicorn_ssl_kwargs_requests_client_cert(self, tmp_path, monkeypatch):
        """When TLS is on + a CA bundle exists, listener requests client certs."""
        import ssl

        from forgather.tls import uvicorn_ssl_kwargs
        from forgather.tls.config import load_config, save_config

        (tmp_path / "server.crt").write_text("dummy cert\n")
        (tmp_path / "server.key").write_text("dummy key\n")
        (tmp_path / "ca-bundle.crt").write_text("dummy bundle\n")
        monkeypatch.setenv("FORGATHER_TLS_DIR", str(tmp_path))
        cfg = load_config()
        cfg.enabled = True
        save_config(cfg)

        kwargs = uvicorn_ssl_kwargs()
        assert kwargs["ssl_cert_reqs"] == ssl.CERT_OPTIONAL
        assert kwargs["ssl_ca_certs"].endswith("ca-bundle.crt")
        assert kwargs["ssl_certfile"].endswith("server.crt")
        assert kwargs["ssl_keyfile"].endswith("server.key")

    def test_uvicorn_ssl_kwargs_off_returns_empty(self, tmp_path, monkeypatch):
        from forgather.tls import uvicorn_ssl_kwargs

        monkeypatch.setenv("FORGATHER_TLS_DIR", str(tmp_path))
        assert uvicorn_ssl_kwargs() == {}


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
