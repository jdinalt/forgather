"""Browser → spawned-TensorBoard reverse proxy.

The forgather server spawns ``tensorboard`` subprocesses bound to
loopback (the new ``tensorboard_ops`` default — see C3). On a multi-user
host that's necessary: TB has no auth of its own, and TB job dirs may
contain training metadata other local users on the box shouldn't see.

But loopback-only also means the browser can't talk to TB directly —
the webui talks to the forgather server, not to the TB port. This
module bridges the gap with an auth-gated reverse proxy mounted under
``/api/tb/{job_id}/{path:path}``. The forgather server's auth
middleware gates the whole ``/api/`` tree, so reaching TB now requires
the same bearer token / session cookie / query token a normal API
request needs.

For the TB UI to stay self-consistent, TB must generate links that use
the proxy's mount path. We pass ``--path_prefix /api/tb/<queue_id>``
when spawning the subprocess (set by the scheduler / launcher); the
proxy here strips that same prefix on inbound requests and forwards the
remainder upstream. Requests / response bodies are streamed
byte-for-byte; redirects are passed back unchanged so TB's own
client-side routing stays correct.

WebSockets are not proxied in this iteration — the TB feature surface
the webui surfaces (scalars, images, hparams, projector) is HTTP-only.
A user who needs the realtime profile plugin can fall back to
``bind_all=True`` on a trusted single-user host.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .. import job_records

log = logging.getLogger("forgather_server.tb_proxy")

router = APIRouter(tags=["tb-proxy"])

# TB responses can be large (event-file dumps, image grids), but they
# don't stream indefinitely — bound the connect/write but leave reads
# uncapped so a slow TB plugin doesn't get cut off mid-response.
_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

_STREAM_CHUNK = 64 * 1024

# Hop-by-hop headers per RFC 7230 §6.1; never forwarded across proxies.
# ``Authorization`` is dropped so a bearer token authenticating the
# browser→forgather hop never leaks into the loopback hop (TB doesn't
# care, but defence in depth). ``host`` is dropped because httpx will
# set its own based on the upstream URL.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "authorization",
        "cookie",
        # Content-Length is recomputed by httpx on send; if we forward
        # the original it can disagree with the body httpx assembles.
        "content-length",
    }
)


def _filter_request_headers(items: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for k, v in items:
        if k.lower() in _HOP_BY_HOP:
            continue
        out.append((k, v))
    return out


def _filter_response_headers(items: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for k, v in items:
        lk = k.lower()
        if lk in _HOP_BY_HOP:
            continue
        # Content-Length is set by Starlette / httpx based on the body
        # we actually emit; the upstream value can be wrong after
        # transcoding (e.g. compression negotiation) and confuses some
        # browsers if it disagrees with framing.
        if lk == "content-length":
            continue
        out.append((k, v))
    return out


def _lookup_upstream(job_id: str) -> Tuple[str, int]:
    """Resolve a TB ``queue_id`` to its loopback ``(host, port)``.

    404s for an unknown id, a non-TB job, or a job that has already
    terminated (its port is no longer bound). The "terminal status" check
    is a courtesy — a request that races a kill will still get a 502 from
    the upstream-connect layer below.
    """
    rec = job_records.get_record(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    if rec.job_type != "tensorboard":
        raise HTTPException(status_code=404, detail="not a tensorboard job")
    if rec.status in job_records.TERMINAL_STATUSES:
        raise HTTPException(status_code=404, detail="tensorboard job has exited")
    port = rec.job_params.get("port") if rec.job_params else None
    if port is None:
        raise HTTPException(status_code=500, detail="tensorboard job has no port")
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        raise HTTPException(status_code=500, detail="tensorboard job port is not int")
    # Even if the user opted into ``bind_all`` / non-loopback host, the
    # proxy itself always reaches TB over loopback — TB binds the
    # loopback interface alongside whatever else, and 127.0.0.1 is the
    # safest dial-out target from the server process.
    return ("127.0.0.1", port_int)


# All HTTP methods TB might receive (mostly GET, but a few internal
# state endpoints accept POST for plugin config). HEAD/OPTIONS round
# out the standard set so curl users get sensible behaviour too.
@router.api_route(
    "/tb/{job_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
)
async def tb_proxy(job_id: str, path: str, request: Request):
    host, port = _lookup_upstream(job_id)

    # TB was launched with ``--path_prefix /api/tb/<job_id>``, so its
    # internal links already include that prefix; preserve it on the
    # upstream URL too.
    upstream_path = f"/api/tb/{job_id}/{path}" if path else f"/api/tb/{job_id}"
    upstream_url = f"http://{host}:{port}{upstream_path}"

    method = request.method
    body = await request.body()
    upstream_headers = _filter_request_headers(request.headers.items())

    # Preserve the original query string verbatim — TB's URLs include
    # selectors like ``?tagFilter=loss&run=run-1`` that depend on
    # untouched encoding to round-trip through TB's plugin manager.
    raw_qs = request.url.query
    params = raw_qs if raw_qs else None

    client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False)
    try:
        req = client.build_request(
            method,
            upstream_url,
            content=body if body else None,
            headers=upstream_headers,
            params=params,
        )
        try:
            response = await client.send(req, stream=True)
        except httpx.RequestError as e:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    except HTTPException:
        raise
    except Exception:
        await client.aclose()
        raise

    response_headers = _filter_response_headers(response.headers.items())

    # Small / non-streaming responses (3xx redirects, error pages) get
    # buffered so we can release the upstream connection promptly. Larger
    # responses fall through to the streaming path below.
    if (
        response.status_code >= 300 and response.status_code < 400
    ) or response.status_code >= 400:
        try:
            content = await response.aread()
        finally:
            await response.aclose()
            await client.aclose()
        return Response(
            content=content,
            status_code=response.status_code,
            headers=dict(response_headers),
        )

    async def body_iter():
        try:
            async for chunk in response.aiter_raw(chunk_size=_STREAM_CHUNK):
                if chunk:
                    yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    media_type = response.headers.get("content-type")
    return StreamingResponse(
        body_iter(),
        status_code=response.status_code,
        headers=dict(response_headers),
        media_type=media_type,
    )
