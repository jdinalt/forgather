"""Bearer-token auth on the inference server's FastAPI app.

Drives ``create_app(auth_token=...)`` directly via FastAPI's TestClient
so the model loader / generation path is bypassed — we only care about
auth gating and which routes are public.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Importing the server package as ``inference_server.routes`` requires
# tools/ on sys.path. conftest.py already prepends the parent dir but
# routes.py uses relative imports, so we explicitly fix up here.
THIS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = THIS_DIR.parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from inference_server.routes import create_app, set_inference_service  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_service():
    # Routes branch on inference_service is None to short-circuit before
    # invoking the model. Leaving it as None is what we want here — auth
    # runs first, so 401 lands before the 500 ever matters.
    set_inference_service(None)
    yield
    set_inference_service(None)


def test_health_open_when_auth_enabled():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_health_open_when_auth_disabled():
    app = create_app(auth_token=None)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_models_requires_auth_when_enabled():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers
    assert "Bearer" in r.headers["WWW-Authenticate"]


def test_models_rejects_wrong_token():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.get("/v1/models", headers={"Authorization": "Bearer wrongtoken"})
    assert r.status_code == 401


def test_models_rejects_non_bearer_scheme():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.get("/v1/models", headers={"Authorization": "Basic secrettoken"})
    assert r.status_code == 401


def test_models_accepts_valid_bearer():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.get("/v1/models", headers={"Authorization": "Bearer secrettoken"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"


def test_models_accepts_when_no_auth():
    app = create_app(auth_token=None)
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200


def test_chat_completions_requires_auth():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_completions_requires_auth():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.post("/v1/completions", json={"model": "test", "prompt": "hi"})
    assert r.status_code == 401


def test_tokenize_requires_auth():
    app = create_app(auth_token="secrettoken")
    client = TestClient(app)
    r = client.post("/tokenize", json={"prompt": "hi"})
    assert r.status_code == 401
    r = client.post("/v1/tokenize", json={"prompt": "hi"})
    assert r.status_code == 401


def test_bearer_compare_constant_time_equiv():
    """Confirm the dependency uses an equality check that won't 401 a
    correct token. Doesn't measure timing — only that a long token with
    a single-char difference still rejects, exercising compare_digest."""
    app = create_app(auth_token="a" * 64)
    client = TestClient(app)
    r = client.get("/v1/models", headers={"Authorization": "Bearer " + "a" * 63 + "b"})
    assert r.status_code == 401
