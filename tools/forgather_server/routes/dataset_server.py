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
registry — and, because ``POST /api/dataset-servers/user`` is gated by
the same bearer, it can register a fresh URL first and then proxy to
it. Concretely, that gives a stolen bearer "make GET / POST requests
to any HTTP host the forgather-server can reach, with attacker-chosen
Authorization header and JSON body, but only against the
``/v1/{health,auth-status,datasets,cache,local,load,length,iter}``
path set." That's bounded but not zero: paths like ``/v1/load`` accept
an attacker-supplied body, so a stolen bearer is effectively
LAN-wide-HTTP-POST capability against the targeted path set.

The forgather-server bearer is already documented as a uid-level
credential (see ``tools/forgather_server/README.md``), so this is a
small amplification of an already-broad threat rather than a new
class. The audit story: registry adds are durable, persisted in
``<config>/server/dataset_server_registry.json``, and visible via
``GET /api/dataset-servers/user`` — unfamiliar entries indicate
compromise.

Token resolution: explicit ``X-Dataset-Auth-Token`` header → JobRecord
auto-lookup (for forgather_server-spawned instances) → registry lookup
(for user-added entries) → none.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

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


class BundleResponse(BaseModel):
    """A self-describing ``forgather-dataset://`` URI that bundles a
    dataset_server's base URL and bearer token for transfer to another
    machine. The destination machine's "+ Add" modal can paste this
    into a single field and decode both halves at once — fewer
    copy-paste steps for the operator, and the token never appears
    separately in clipboard history.

    Equivalent to an SSH private key on the wire: anyone who sees the
    bundle gets full client access to the named server. Treat it
    accordingly.
    """

    bundle: str


@router.get("/dataset-servers/local/{queue_id}/bundle", response_model=BundleResponse)
def local_server_bundle(queue_id: str):
    """Mint a transferable bundle for a locally-spawned dataset_server.

    Only available for forgather_server-launched instances (token comes
    from the JobRecord). User-added entries already know their own
    token on the source machine — no server round-trip needed.
    """
    for r in job_records.list_records():
        if r.job_type != "dataset_server" or r.queue_id != queue_id:
            continue
        if r.status not in {"starting", "running"}:
            raise HTTPException(
                status_code=410,
                detail=f"dataset_server {queue_id} is not running",
            )
        params = r.job_params or {}
        port = params.get("port")
        try:
            port = int(port) if port is not None else None
        except (TypeError, ValueError):
            port = None
        if port is None:
            raise HTTPException(status_code=500, detail="job has no port in job_params")
        host = params.get("host") or "127.0.0.1"
        # Bundle host: same translation the proxy/JobsPanel use.
        # 0.0.0.0 binds everywhere but isn't a routable client target;
        # default to "localhost" so the destination machine can replace
        # it with whatever hostname they actually reach this box on.
        bundle_host = "localhost" if host == "0.0.0.0" else host
        token = r.auth_token or ""
        # urlencode the token defensively — bearer tokens are hex today
        # but the URI shape shouldn't assume it.
        bundle = (
            f"forgather-dataset://{bundle_host}:{port}"
            f"/?token={quote(token, safe='')}"
        )
        return BundleResponse(bundle=bundle)
    raise HTTPException(status_code=404, detail=f"no local dataset_server: {queue_id}")


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
    try:
        entry = dataset_server_registry.add_entry(
            label=req.label,
            base_url=base_url,
            auth_token=(req.auth_token or "").strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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


# --- handle-level proxy ---
#
# The browser-facing endpoints below let the Explore tab load a dataset
# on demand and page through its rows. The dataset_server's own /iter
# returns NDJSON (one example per line, streamed); we collect a bounded
# window into a JSON list so the webui doesn't have to wire up a line
# parser. Page-size caps are enforced upstream by the limit query
# parameter — anything that fits in a single response is small enough
# to materialize here.


@router.post("/dataset-server/proxy/load")
async def proxy_load(base: str, request: Request):
    """Forward POST /v1/load. Body is a JSON object passed through."""
    target = _validate_base(base) + "/v1/load"
    try:
        body = await request.body()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read body: {e}")
    headers = _auth_headers_for(base, request)
    headers["content-type"] = request.headers.get("content-type", "application/json")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.post(target, content=body, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


@router.get("/dataset-server/proxy/length")
async def proxy_length(base: str, handle: str, request: Request):
    """Forward GET /v1/datasets/{handle}/length."""
    # URL-encode the handle so a malformed value (slashes, control
    # chars) can't escape its path segment. Defense in depth — the
    # upstream's own router would reject the malformed handle, but
    # belt-and-suspenders is cheap.
    return await _proxy_get(
        base, f"/v1/datasets/{quote(handle, safe='')}/length", request
    )


@router.get("/dataset-server/proxy/iter")
async def proxy_iter(
    base: str,
    handle: str,
    request: Request,
    position: int = 0,
    limit: int = 25,
    seed: Optional[int] = None,
):
    """Forward GET /v1/datasets/{handle}/iter and materialize the NDJSON
    stream into ``{"rows": [...]}``.

    ``limit`` is capped (max 500) because the result is fully buffered
    in memory before returning; the explore tab pages with limit=25 by
    default. Larger pages stay supported for ad-hoc CLI users hitting
    the same proxy.
    """
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    if limit > 500:
        raise HTTPException(status_code=400, detail="limit must be <= 500")
    if position < 0:
        raise HTTPException(status_code=400, detail="position must be >= 0")

    qs = f"?position={int(position)}&limit={int(limit)}"
    if seed is not None:
        qs += f"&seed={int(seed)}"
    target = _validate_base(base) + f"/v1/datasets/{quote(handle, safe='')}/iter" + qs

    headers = _auth_headers_for(base, request)
    rows: List[Any] = []
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            async with client.stream("GET", target, headers=headers or None) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    try:
                        detail = body.decode("utf-8", errors="replace")
                    except Exception:
                        detail = "<binary>"
                    return JSONResponse(
                        status_code=r.status_code,
                        content={"detail": detail},
                        headers=_upstream_auth_headers(r.status_code),
                    )
                # NDJSON: one JSON value per line. iter_lines handles
                # both \n and \r\n; we ignore blanks so a trailing
                # newline doesn't surface as a None row.
                import json as _json

                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        rows.append(_json.loads(line))
                    except ValueError:
                        # Don't drop the whole window for one bad line —
                        # surface as a string so the user can at least see
                        # something went wrong.
                        rows.append({"_parse_error": line})
                    # Defensive cap: the upstream is asked for ``limit``
                    # via the query string, but a misbehaving server
                    # could ignore that and stream arbitrarily many
                    # rows. Stop reading once we've buffered the
                    # caller-requested cap so the proxy can't be
                    # turned into a memory DoS by a hostile upstream.
                    if len(rows) >= limit:
                        break
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse({"rows": rows})
