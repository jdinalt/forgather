"""Dataset-server convenience endpoints.

Two responsibilities:

1. ``POST /dataset-server/config/ensure-stub`` — used by the Tools menu's
   right-click "Edit Configuration…" item. Returns the absolute path of
   the default config, creating a commented stub if it doesn't exist.
2. The Datasets view's *Servers* tab. Lets the webui CRUD a small
   registry of user-added dataset_server URLs (URL + token), and proxies
   read-only metadata calls (``/v1/health``, ``/v1/datasets``,
   ``/v1/cache/hf``, ``/v1/local``) so the browser doesn't have to talk
   to each token-gated upstream directly.

Path used by (1): ``<forgather_config_dir>/dataset_server/config.yaml``
— same directory the standalone dataset_server reads from when
``--config`` is omitted.

SSRF policy
-----------
Unlike the inference proxy (where remote targets are exceptional), the
dataset_server is *expected* to live on a different host: the primary
deployment shape is one dataset host serving several training nodes
across a LAN. A "localhost only" default would just push every operator
into setting an env var or, worse, running with ``--no-auth`` — neither
helps security.

Instead, the proxy uses the user registry itself as the SSRF allowlist:

  - Loopback hosts (``127.0.0.1`` / ``localhost`` / ``::1``) are always
    allowed — that covers same-host development and the auto-discovered
    JobRecord-spawned instances.
  - Any URL the operator has registered via ``POST /api/dataset-servers/user``
    is allowed. The act of registering an entry *is* the explicit
    authorization decision; "I added this URL to my dataset-server list"
    is exactly the consent the SSRF gate needs.
  - Everything else returns 403 with a hint to register the URL first.

A stolen forgather-server bearer can still proxy to anything in the
registry — but the registry is auditable, and registering a fresh URL
takes a separate, explicit API call. Compared to either "always allow"
or the inference-style env-var gate, this gives the operator the same
explicit control without an obscure flag they have to discover.

Token resolution: explicit ``X-Dataset-Auth-Token`` header → JobRecord
auto-lookup (for forgather_server-spawned instances) → registry lookup
(for user-added entries) → none.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from forgather.preprocess import forgather_config_dir

from .. import dataset_server_registry, job_records

log = logging.getLogger("forgather_server.routes.dataset_server")
router = APIRouter(tags=["dataset_server"])


# Stub content for a freshly-created config.yaml. Mirrors the YAML
# example in tools/dataset_server/README.md; every line is commented
# so the file is "valid YAML with no overrides" until the user
# actively edits in their settings.
_STUB_CONFIG = """\
# Forgather dataset_server configuration
#
# This file is loaded automatically when `forgather dataset-server start`
# is run without `--config`. CLI flags always override file values.
# See tools/dataset_server/README.md for the full reference.

# Bind address + port. Loopback by default; switch to 0.0.0.0 for LAN.
# host: 127.0.0.1
# port: 8766
# log_level: INFO

# Auth (mutually exclusive; default = auto-generated per-port token).
# Setting `no_auth: true` disables the bearer-token gate entirely —
# only do this on a trusted network.
# no_auth: false
# auth_token_file: ~/.fdss.token

# Loading policy — all default to the safe option.
# no_hf: false              # disable HF cache loading (local/* only)
# allow_paths: false        # allow loads by absolute filesystem path
# allow_downloads: false    # allow HF downloads on cache miss

# Named local datasets. Clients request these as `local/<name>`.
# Paths must exist at server startup.
# local:
#   stories: /data/tinystories
#   mycorpus: /data/saved_corpus
"""


def _default_config_path() -> Path:
    """Same path the dataset_server itself reads at startup.

    Kept in sync with ``tools/dataset_server/server.py::default_config_file``.
    Duplicated rather than imported so the forgather_server package
    doesn't take a hard dependency on the dataset_server entry-point
    module's sys.path shenanigans.
    """
    return Path(forgather_config_dir()) / "dataset_server" / "config.yaml"


class EnsureStubResponse(BaseModel):
    path: str
    created: bool


@router.post("/dataset-server/config/ensure-stub", response_model=EnsureStubResponse)
def ensure_stub() -> EnsureStubResponse:
    """Return the absolute path of the default config; create stub if absent.

    The stub is created with 0600 perms inside a 0700 directory — same
    tightening the rest of the dataset_server's persistent state uses,
    since the file may eventually contain an auth_token_file path or
    other operator-sensitive content.
    """
    path = _default_config_path()
    if path.is_file():
        return EnsureStubResponse(path=str(path), created=False)

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError as e:
        log.warning("could not chmod %s to 0700: %s", parent, e)

    # O_EXCL guards against a TOCTOU race: two concurrent webui clicks
    # would otherwise both hit the "not is_file()" branch and one would
    # clobber the other's just-written file. EEXIST means somebody else
    # won; treat that as "already there".
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError:
        return EnsureStubResponse(path=str(path), created=False)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_STUB_CONFIG)
    except Exception:
        # Best-effort cleanup: if the write fails partway through we
        # don't want to leave an empty / half-written file that the
        # editor opens with no content.
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    log.info("created stub dataset_server config at %s", path)
    return EnsureStubResponse(path=str(path), created=True)


# ---------------------------------------------------------------------------
# Server discovery + user-added registry
# ---------------------------------------------------------------------------


class LocalServerModel(BaseModel):
    """A dataset_server instance launched by this forgather_server.

    ``base_url`` is the loopback URL the browser can reach via the proxy.
    ``has_auth_token`` is exposed so the webui can render an "auth ✓"
    indicator without ever shipping the token to the client (the proxy
    handles the actual auth header). Operators can still see the token
    on the Jobs panel where it's already surfaced.
    """

    queue_id: str
    label: str
    base_url: str
    host: str
    port: int
    alive: bool
    has_auth_token: bool


class UserEntryModel(BaseModel):
    id: str
    label: str
    base_url: str
    has_auth_token: bool


class AddUserEntryRequest(BaseModel):
    label: str = ""
    base_url: str
    auth_token: str = ""


def _local_servers() -> List[LocalServerModel]:
    """Enumerate dataset_server jobs we spawned. Mirrors the inference
    proxy's JobRecord scan but only surfaces user-facing metadata —
    auth tokens stay server-side.
    """
    out: List[LocalServerModel] = []
    for r in job_records.list_records():
        if r.job_type != "dataset_server":
            continue
        params = r.job_params or {}
        port = params.get("port")
        try:
            port = int(port) if port is not None else None
        except (TypeError, ValueError):
            port = None
        if port is None:
            continue
        host = params.get("host") or "127.0.0.1"
        # 0.0.0.0 binds everywhere but the browser-facing URL should
        # still resolve cleanly — same translation the Jobs view does.
        browser_host = "localhost" if host == "0.0.0.0" else host
        base_url = f"http://{browser_host}:{port}"
        alive = r.status in {"starting", "running"}
        out.append(
            LocalServerModel(
                queue_id=r.queue_id,
                label=f"{r.config or 'dataset_server'}:{port}",
                base_url=base_url,
                host=str(host),
                port=port,
                alive=alive,
                has_auth_token=bool(r.auth_token),
            )
        )
    # Newest first — matches the Jobs view's implicit ordering and means
    # a freshly-spawned server is at the top where the user expects.
    out.sort(key=lambda s: s.queue_id, reverse=True)
    return out


@router.get("/dataset-servers/local", response_model=List[LocalServerModel])
def list_local_servers():
    return _local_servers()


@router.get("/dataset-servers/user", response_model=List[UserEntryModel])
def list_user_entries():
    return [
        UserEntryModel(
            id=e.id,
            label=e.label,
            base_url=e.base_url,
            has_auth_token=bool(e.auth_token),
        )
        for e in dataset_server_registry.list_entries()
    ]


@router.post("/dataset-servers/user", response_model=UserEntryModel)
def add_user_entry(req: AddUserEntryRequest):
    base_url = (req.base_url or "").strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")
    try:
        parsed = urlparse(base_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base_url: {e}")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported scheme: {parsed.scheme!r}",
        )
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="bad base_url: missing host")
    entry = dataset_server_registry.add_entry(
        label=req.label,
        base_url=base_url,
        auth_token=(req.auth_token or "").strip(),
    )
    return UserEntryModel(
        id=entry.id,
        label=entry.label,
        base_url=entry.base_url,
        has_auth_token=bool(entry.auth_token),
    )


@router.delete("/dataset-servers/user/{entry_id}")
def delete_user_entry(entry_id: str):
    removed = dataset_server_registry.remove_entry(entry_id)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"no entry: {entry_id}")
    return {"removed": removed.id}


# ---------------------------------------------------------------------------
# Read-only metadata proxy
# ---------------------------------------------------------------------------

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

_TOKEN_OVERRIDE_HEADER = "x-dataset-auth-token"
_UPSTREAM_AUTH_FAILED_HEADER = "x-upstream-auth-failed"


def _registered_base_urls() -> frozenset[str]:
    """Snapshot of base URLs in the user registry, normalized.

    URLs are stored without trailing slash; we match by exact string
    equality after the same normalization, so an entry registered as
    ``http://datahost:8766`` matches a proxy request for
    ``http://datahost:8766`` but not for ``http://datahost:8766/``
    (the proxy callers normalize too — see ``_validate_base``).
    """
    return frozenset(e.base_url for e in dataset_server_registry.list_entries())


def _validate_base(base: str) -> str:
    """SSRF policy: loopback always, registered URLs always, otherwise 403.

    The user registry doubles as the host-allowlist for the proxy. A
    URL the operator has explicitly added via ``POST
    /api/dataset-servers/user`` is treated as authorized; anything else
    that isn't loopback is rejected with a clear "register first" hint.
    """
    try:
        parsed = urlparse(base)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base url: {e}")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported scheme: {parsed.scheme!r}",
        )
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="missing host")
    normalized = base.rstrip("/")
    host = (parsed.hostname or "").lower()
    if host in _LOCALHOST_HOSTS:
        return normalized
    if normalized in _registered_base_urls():
        return normalized
    raise HTTPException(
        status_code=403,
        detail=(
            f"refusing to proxy to unregistered host: {host!r}. "
            "Add it via the Datasets view (+ Add) or POST "
            "/api/dataset-servers/user first."
        ),
    )


def _token_for_local(base: str) -> Optional[str]:
    """Auto-lookup token from JobRecords (matches inference proxy)."""
    try:
        parsed = urlparse(base)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if host not in _LOCALHOST_HOSTS or parsed.port is None:
        return None
    for r in job_records.list_records():
        if r.job_type != "dataset_server":
            continue
        if r.status not in {"starting", "running"}:
            continue
        params = r.job_params or {}
        rec_port = params.get("port")
        try:
            rec_port = int(rec_port) if rec_port is not None else None
        except (TypeError, ValueError):
            continue
        if rec_port != parsed.port:
            continue
        rec_host = (params.get("host") or "127.0.0.1").lower()
        # Treat any-host bind (0.0.0.0) as matching loopback queries —
        # same translation the local-servers list uses.
        if rec_host not in _LOCALHOST_HOSTS and rec_host != "0.0.0.0":
            continue
        return r.auth_token
    return None


def _auth_headers_for(base: str, request: Request) -> Dict[str, str]:
    """Build the upstream auth header dict.

    Precedence: explicit ``X-Dataset-Auth-Token`` (from the webui's
    user-entry rows or any CLI caller that knows the token), then
    JobRecord auto-lookup, then registry lookup for a saved user entry,
    then empty (server is running --no-auth).
    """
    override = request.headers.get(_TOKEN_OVERRIDE_HEADER)
    if override:
        return {"authorization": f"Bearer {override}"}
    token = _token_for_local(base)
    if token:
        return {"authorization": f"Bearer {token}"}
    saved = dataset_server_registry.find_token(base)
    if saved:
        return {"authorization": f"Bearer {saved}"}
    return {}


def _upstream_auth_headers(status: int) -> Dict[str, str]:
    if status in (401, 403):
        return {_UPSTREAM_AUTH_FAILED_HEADER: "1"}
    return {}


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return {"error": "non-json response from upstream", "body": r.text}


async def _proxy_get(base: str, upstream_path: str, request: Request) -> JSONResponse:
    target = _validate_base(base) + upstream_path
    headers = _auth_headers_for(base, request)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.get(target, headers=headers or None)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


@router.get("/dataset-server/proxy/health")
async def proxy_health(base: str, request: Request):
    return await _proxy_get(base, "/v1/health", request)


@router.get("/dataset-server/proxy/auth-status")
async def proxy_auth_status(base: str, request: Request):
    return await _proxy_get(base, "/v1/auth/status", request)


@router.get("/dataset-server/proxy/datasets")
async def proxy_datasets(base: str, request: Request):
    return await _proxy_get(base, "/v1/datasets", request)


@router.get("/dataset-server/proxy/cache")
async def proxy_cache(base: str, request: Request):
    return await _proxy_get(base, "/v1/cache/hf", request)


@router.get("/dataset-server/proxy/local")
async def proxy_local(base: str, request: Request):
    return await _proxy_get(base, "/v1/local", request)
