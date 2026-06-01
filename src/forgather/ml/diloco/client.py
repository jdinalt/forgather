"""
DiLoCo HTTP Client.

Simple HTTP client for communicating with the DiLoCo parameter server.
Uses urllib.request (stdlib) to avoid extra dependencies. Handles tensor
serialization/deserialization via torch.save/load.

Usage:
    client = DiLoCoClient("192.168.1.100:8512")
    global_params = client.register("worker-0", {"hostname": "machine-a"})
    # ... train for H steps ...
    new_params = client.submit_pseudogradients("worker-0", pseudograds)
"""

import io
import json
import logging
import os
import struct
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import torch

from forgather.ml.diloco.auth import read_standalone_token

logger = logging.getLogger(__name__)

#: Env var consulted as a fallback when no explicit token is passed.
TOKEN_ENV_VAR = "FORGATHER_DILOCO_SERVER_TOKEN"


class DiLoCoModelMismatchError(ConnectionError):
    """Raised when ``/register`` returns HTTP 422 for a model fingerprint
    mismatch.

    Workers send a ``param_shapes`` map (``{name: [d0, d1, …]}``) for
    every entry in ``self.model.named_parameters()``. The server
    compares against its own param set + shapes. Mismatches are
    fatal — without this check, a wrong ``--model-id-or-path`` on a
    worker would silently train a different architecture against the
    server's global weights and crash hundreds of steps later in the
    first sync's optimizer step. Same failure mode as #45 but caused
    by operator misconfiguration rather than save/load mechanics.

    Inherits from ConnectionError so legacy broad-catch callers still
    work; new callers can branch on the specific type.
    """

    def __init__(self, message: str, *, diagnostic: str = ""):
        super().__init__(message)
        self.diagnostic = diagnostic


class DiLoCoServerUnreachable(ConnectionError):
    """Raised when the DiLoCo server can't be contacted at startup.

    The DiLoCoCallback does a ``/status`` round-trip in
    ``on_train_begin`` before constructing the worker; if the server
    URL is wrong, the server is down, or a firewall is in the way,
    we fail loudly here rather than letting training begin and then
    diverging silently when the first sync 500 steps later fails.

    Inherits from ConnectionError so callers catching the broader
    type still work; new callers can branch on the specific type.
    """

    pass


class DiLoCoRegisterCollisionError(ConnectionError):
    """Raised when ``/register`` returns HTTP 409.

    The server enforces ``worker_id`` uniqueness (see
    docs/design/diloco-work-unit-dispatch.md): a second registration of
    an already-live ``worker_id`` is refused with 409 + a diagnostic.
    Workers that catch this should treat it as a fatal clean-exit
    (re-registering won't succeed until the prior entry is evicted by
    heartbeat timeout or cleared via /deregister).

    Inherits from ConnectionError so legacy callers that catch the
    broader exception type continue to work. New callers can match the
    specific type to branch on the collision case.
    """

    def __init__(self, message: str, *, diagnostic: str = ""):
        super().__init__(message)
        # The server's diagnostic body (e.g. the "worker_id 'X' is
        # already registered…" string) — surfaced separately so
        # callers can log it cleanly without re-parsing the message.
        self.diagnostic = diagnostic


class DiLoCoClient:
    """
    HTTP client for DiLoCo parameter server communication.

    Handles tensor serialization (via torch.save to BytesIO), request
    construction, and response parsing. All methods are synchronous/blocking.

    Args:
        server_addr: Server address as "host:port" (e.g., "192.168.1.100:8512").
        timeout: Request timeout in seconds. Pseudo-gradient submission may
            block for a long time waiting for the sync barrier, so this should
            be generous.
        max_retries: Maximum retries for transient failures.
        retry_delay: Base delay between retries (seconds). Doubles each retry.
    """

    def __init__(
        self,
        server_addr: str,
        timeout: float = 600,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        token: Optional[str] = None,
        verify_tls: bool = True,
    ):
        # Normalize address. A bare ``host:port`` is the legacy form
        # (the ``forgather diloco worker --server host:port`` CLI and
        # pre-#90 callers all hand off this shape). Pick the scheme
        # the same way the scheduler does for its JobRecord
        # ``scheme`` stamp — ``forgather.tls.client_scheme()`` —
        # which returns ``"https"`` when TLS is provisioned locally,
        # else ``"http"``. That keeps the worker in sync with the
        # server's actual posture for the trusted-LAN case where
        # both ends share the same TLS provisioning.
        # Whether we *guessed* the scheme (bare host:port) vs. it being
        # explicit in the URL. A guessed scheme is derived from THIS
        # host's TLS provisioning, which need not match the server's —
        # a TLS-off worker pointed at a TLS server guesses http:// and
        # the TLS handshake surfaces as a bare connection reset. We
        # remember the guess so connection failures can name this as the
        # likely cause instead of failing silently/cryptically.
        self._scheme_guessed = not server_addr.startswith(("http://", "https://"))
        if self._scheme_guessed:
            # Loopback servers default to http:// (the long-standing
            # default, and what local dev / test servers actually
            # speak). Only non-loopback hosts consult client_scheme():
            # a routable bare host:port — e.g. a worker handed
            # "192.168.9.43:8512" — picks https when this host has TLS
            # provisioned, which is the trusted-LAN case commit
            # 72c3225a fixed. Inferring https for loopback would break
            # every cleartext localhost server whenever the operator
            # happens to have a TLS config on disk.
            bare_host = (
                server_addr.rsplit(":", 1)[0] if ":" in server_addr else server_addr
            )
            bare_host = bare_host.strip("[]")  # ipv6 literal
            scheme = "http"
            try:
                from forgather.tls.policy import host_is_loopback

                if not host_is_loopback(bare_host):
                    from forgather.tls import client_scheme as _client_scheme

                    scheme = _client_scheme(bare_host)
            except Exception:
                scheme = "http"
            server_addr = f"{scheme}://{server_addr}"
        self.server_addr = server_addr.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Security (issue #90): bearer token + TLS verification.
        # Token resolution precedence (mirrors dataset_server's
        # RemoteBackend): explicit ``token=`` arg → env var
        # ``FORGATHER_DILOCO_SERVER_TOKEN`` → per-port loopback file.
        # Returns ``None`` for remote URLs without an explicit token,
        # in which case requests go unauthenticated (and will 401 if
        # the server has auth enabled — the caller sees a clean
        # ``ConnectionError`` describing the 401).
        if token is None:
            token = os.environ.get(TOKEN_ENV_VAR) or None
        if token is None:
            token = read_standalone_token(self.server_addr)
        self.token = token
        self.verify_tls = verify_tls
        # SSL context for the control URL is cached. Bulk-port URLs
        # (learned at /register time) may have a different scheme, so
        # contexts for those are resolved per-request via ``_ssl_for``.
        self._ssl_ctx = self._ssl_for(self.server_addr)
        # ``bulk_url`` is populated from the ``X-Forgather-Bulk-Url``
        # response header on /register (issue #90). When set, the
        # three bulk endpoints (submit_pseudograd,
        # submit_fragment_pseudograd, global_params) route to it
        # instead of the control URL. ``None`` keeps the single-port
        # behavior — bulk endpoints go to the same listener.
        self.bulk_url: Optional[str] = None
        # Cached SSL context for the bulk URL (built lazily on first
        # bulk request; invalidated when the URL changes on
        # reconnect). Explicit None here so the attribute exists from
        # construction time — keeps ``__slots__`` migrations and static
        # type checkers happy.
        self._bulk_ssl_ctx: Optional["ssl.SSLContext"] = None

    def _scheme_hint(self) -> str:
        """Diagnostic suffix for connection errors when the URL scheme
        was inferred rather than explicit.

        A bare ``host:port`` worker derives http vs https from *its own*
        TLS provisioning, which may not match the server's. When that
        guess is wrong the failure is a low-level connection reset with
        no obvious cause; this spells it out and tells the operator how
        to remove the ambiguity."""
        if not getattr(self, "_scheme_guessed", False):
            return ""
        scheme = self.server_addr.split("://", 1)[0]
        other = "http" if scheme == "https" else "https"
        return (
            f" [scheme {scheme}:// was inferred from this host's local TLS "
            f"config, not the server's — if the server actually speaks "
            f"{other}://, this is the expected symptom; pass an explicit "
            f"{other}://host:port server address]"
        )

    def _ssl_for(self, url: str) -> Optional["ssl.SSLContext"]:
        """SSL context appropriate for ``url`` — None for ``http://``.

        Built once per scheme; we keep a cached one for the control
        URL and build a fresh one for the bulk URL the first time we
        see it. Cheap to build (no network I/O) so caching isn't
        critical, but it keeps cert-chain parsing off the hot path.
        """
        if not url.lower().startswith("https"):
            return None
        from forgather.tls.runtime import urllib_ssl_context

        return urllib_ssl_context(verify=self.verify_tls)

    def _ssl_for_request(self, url: str) -> Optional["ssl.SSLContext"]:
        """Return the cached control-port SSL context when ``url``
        matches it; otherwise build/cache a fresh one for the URL.

        Cleartext URLs short-circuit to ``None``. The cache is a
        single-entry slot keyed by the bulk URL, since a worker only
        ever talks to one bulk listener per server.
        """
        if not url.lower().startswith("https"):
            return None

        def _same_base(u: str, base: Optional[str]) -> bool:
            # Match on the origin boundary, not a bare prefix: ``base``
            # followed by ``/`` (or exactly equal). Avoids mis-matching
            # when one base is a string prefix of the other — e.g.
            # control ``:8512`` vs bulk ``:85120`` — which a plain
            # ``startswith`` would conflate.
            return bool(base) and (u == base or u.startswith(base + "/"))

        if _same_base(url, self.server_addr):
            return self._ssl_ctx
        if _same_base(url, self.bulk_url):
            if self._bulk_ssl_ctx is None:
                self._bulk_ssl_ctx = self._ssl_for(self.bulk_url)
            return self._bulk_ssl_ctx
        return self._ssl_for(url)

    # Bulk paths that, when ``self.bulk_url`` is populated, route to
    # the bulk listener instead of the control URL (issue #90).
    _BULK_PATHS = frozenset(
        {"/submit_pseudograd", "/submit_fragment_pseudograd", "/global_params"}
    )

    def _base_for_path(self, path: str) -> str:
        """Pick the base URL (control vs bulk) for ``path``."""
        canonical = "/" + path.lstrip("/")
        if self.bulk_url and canonical in self._BULK_PATHS:
            return self.bulk_url
        return self.server_addr

    def _routes_to_bulk(self, path: str) -> bool:
        """True when ``path`` is served on the (cleartext, unauthenticated)
        bulk listener rather than the control plane."""
        canonical = "/" + path.lstrip("/")
        return bool(self.bulk_url) and canonical in self._BULK_PATHS

    def _url(self, path: str) -> str:
        """Build full URL for an endpoint, routing bulk paths to the
        bulk listener when one has been advertised."""
        base = self._base_for_path(path)
        return f"{base}/{path.lstrip('/')}"

    def _headers(
        self, content_type: Optional[str] = None, *, path: Optional[str] = None
    ) -> Dict[str, str]:
        """Build request headers, attaching the bearer token when known.

        ``Authorization: Bearer <token>`` is sent on control-plane
        requests when a token is configured. The server verifies it via
        constant-time compare (see ``ml/diloco/auth.py``).

        It is deliberately **omitted** for requests routed to the bulk
        listener (``path`` in ``_BULK_PATHS`` with ``bulk_url`` set): that
        plane is cleartext and unauthenticated by design, and the server
        never checks the bearer there. Sending the control-plane token
        over the unencrypted bulk socket would leak full control-plane
        authority to any LAN sniffer — the exact "host takeover" boundary
        the two-plane split exists to hold.
        """
        headers: Dict[str, str] = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if self.token and not (path is not None and self._routes_to_bulk(path)):
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _serialize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        """Serialize a state dict to bytes."""
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return buf.getvalue()

    def _deserialize_state_dict(self, data: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize bytes to a state dict."""
        buf = io.BytesIO(data)
        return torch.load(buf, map_location="cpu", weights_only=True)

    def _request_json(
        self,
        method: str,
        path: str,
        data: Optional[dict] = None,
        retries: Optional[int] = None,
    ) -> dict:
        """Make a JSON request and return parsed JSON response."""
        url = self._url(path)
        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers=self._headers("application/json" if body else None, path=path),
        )

        max_retries = retries if retries is not None else self.max_retries
        delay = self.retry_delay

        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(
                    req,
                    timeout=self.timeout,
                    context=self._ssl_for_request(url),
                ) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                # Server-side 4xx/5xx — the response is the server's
                # authoritative answer. Retrying wouldn't change it.
                # 409 in particular is the work-queue's
                # length-mismatch / worker_id-collision signal and
                # the caller needs to see it promptly. Surface the
                # status code + the server's diagnostic body so the
                # caller can branch on "HTTP 409" cleanly.
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                    error_detail = json.loads(error_body).get("error", error_body)
                except Exception:
                    error_detail = str(e)
                raise ConnectionError(
                    f"Server returned HTTP {e.code} for {url}: {error_detail}"
                ) from e
            except urllib.error.URLError as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Request to {url} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ConnectionError(
                        f"Failed to connect to DiLoCo server at {url}: {e}"
                        f"{self._scheme_hint()}"
                    ) from e

    def _request_tensor(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        content_type: str = "application/octet-stream",
        retries: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Make a request and return deserialized tensor response.

        Args:
            retries: Number of retries on connection failure. Defaults to 0
                (no retries) for backward compatibility. Set to a positive
                value for fault-tolerant reconnection scenarios.
        """
        url = self._url(path)
        max_retries = retries if retries is not None else 0
        delay = self.retry_delay

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                url,
                data=body,
                method=method,
                headers=self._headers(content_type if body else None, path=path),
            )
            try:
                with urllib.request.urlopen(
                    req,
                    timeout=self.timeout,
                    context=self._ssl_for_request(url),
                ) as resp:
                    data = resp.read()
                    return self._deserialize_state_dict(data)
            except urllib.error.HTTPError as e:
                # HTTP error (4xx/5xx) - read the response body for diagnostics
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                    error_detail = json.loads(error_body).get("error", error_body)
                except Exception:
                    error_detail = str(e)
                raise ConnectionError(
                    f"Server returned HTTP {e.code} for {url}: {error_detail}"
                ) from e
            except urllib.error.URLError as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Tensor request to {url} failed "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ConnectionError(
                        f"Failed to connect to DiLoCo server at {url}: {e}"
                        f"{self._scheme_hint()}"
                    ) from e

    def register(
        self, worker_id: str, worker_info: Optional[dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Register with the server and receive global parameters.

        Args:
            worker_id: Unique worker identifier.
            worker_info: Optional metadata dict (hostname, device info, etc.).

        Returns:
            Global model parameters as a state dict.
        """
        import platform

        info = {
            "worker_id": worker_id,
            "hostname": platform.node(),
            **(worker_info or {}),
        }

        body = json.dumps(info).encode("utf-8")
        url = self._url("/register")

        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers=self._headers("application/json"),
        )

        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    req,
                    timeout=self.timeout,
                    context=self._ssl_for_request(url),
                ) as resp:
                    data = resp.read()
                    # Bulk-listener URL advertised in a response header
                    # (issue #90). When the server offloads bulk paths
                    # to a separate port, all future submit_pseudograd
                    # / global_params calls route to that URL.
                    #
                    # On *reconnect*, the server may have changed its
                    # bulk-listener configuration (or removed it
                    # entirely). Always re-read the header — including
                    # when it's absent — and reset accordingly so a
                    # client that learned a bulk_url before a server
                    # restart doesn't keep dialing a dead URL.
                    bulk_url = resp.headers.get("X-Forgather-Bulk-Url")
                    if bulk_url:
                        bulk_url = bulk_url.strip() or None
                    if bulk_url is not None:
                        # Defense in depth: only honor http(s) bulk URLs.
                        # A misconfigured proxy or compromised server
                        # advertising e.g. ``file://`` or
                        # ``javascript:`` would otherwise route bulk
                        # requests somewhere the worker had no
                        # business going. Bad URLs are dropped with a
                        # WARNING; the worker falls back to the
                        # control URL for bulk requests.
                        try:
                            from urllib.parse import urlparse as _urlparse

                            scheme = _urlparse(bulk_url).scheme.lower()
                        except Exception:
                            scheme = ""
                        if scheme not in ("http", "https"):
                            logger.warning(
                                "Ignoring bulk listener URL with "
                                "unsupported scheme %r: %r",
                                scheme,
                                bulk_url,
                            )
                            bulk_url = None
                    if bulk_url != self.bulk_url:
                        if bulk_url:
                            logger.info(
                                f"Server advertised bulk listener at "
                                f"{bulk_url}; bulk endpoints will route there."
                            )
                        elif self.bulk_url:
                            logger.info(
                                "Server no longer advertises a bulk listener; "
                                "bulk endpoints will use the control URL."
                            )
                        self.bulk_url = bulk_url
                        # Invalidate any cached bulk SSL context.
                        self._bulk_ssl_ctx = None
                    params = self._deserialize_state_dict(data)
                    logger.info(
                        f"Registered with server as {worker_id}, received global params"
                    )
                    return params
            except urllib.error.HTTPError as e:
                # HTTPError is a URLError subclass but the response IS
                # from the server (it just isn't 2xx). Retrying won't
                # change the outcome — 409 in particular is the explicit
                # uniqueness-collision signal and the worker needs to
                # see it promptly. Fast-fail with the status code +
                # server diagnostic body so callers can branch on
                # "HTTP 409" cleanly.
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                    error_detail = json.loads(error_body).get("error", error_body)
                except Exception:
                    error_detail = str(e)
                if e.code == 409:
                    # Type-distinguished so callers can match the
                    # collision case specifically. Still a
                    # ConnectionError subclass for back-compat with
                    # broad exception handlers.
                    raise DiLoCoRegisterCollisionError(
                        f"DiLoCo /register returned HTTP 409: {error_detail}",
                        diagnostic=error_detail,
                    ) from e
                if e.code == 422:
                    # Model fingerprint mismatch: worker's
                    # param_shapes don't agree with the server's
                    # _param_list. Operator likely pointed the
                    # worker at the wrong --model-id-or-path. Surface
                    # the diagnostic loudly so the TTY pane shows
                    # the divergent params.
                    raise DiLoCoModelMismatchError(
                        f"DiLoCo /register returned HTTP 422: {error_detail}",
                        diagnostic=error_detail,
                    ) from e
                raise ConnectionError(
                    f"DiLoCo /register returned HTTP {e.code}: {error_detail}"
                ) from e
            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Registration failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ConnectionError(
                        f"Failed to register with DiLoCo server at {url}: {e}"
                        f"{self._scheme_hint()}"
                    ) from e

    def submit_pseudogradients(
        self, worker_id: str, pseudograds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Submit pseudo-gradients and receive updated global parameters.

        This call blocks until all workers have submitted (synchronous barrier
        on the server side). The timeout should be generous enough to allow
        slower workers to finish their local training steps.

        Args:
            worker_id: Worker identifier.
            pseudograds: Pseudo-gradients (global_params - local_params).

        Returns:
            Updated global model parameters after outer optimizer step.
        """
        # Build request body: length-prefixed JSON header + tensor payload
        header = json.dumps({"worker_id": worker_id}).encode("utf-8")
        tensor_data = self._serialize_state_dict(pseudograds)

        body = struct.pack("!I", len(header)) + header + tensor_data

        params = self._request_tensor("POST", "/submit_pseudograd", body=body)
        return params

    def submit_fragment_pseudogradients(
        self,
        worker_id: str,
        fragment_id: int,
        pseudograds: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Submit pseudo-gradients for a single model fragment.

        Used in streaming DiLoCo mode where the model is split into fragments
        that sync at staggered intervals for communication-computation overlap.

        Args:
            worker_id: Worker identifier.
            fragment_id: Which fragment these pseudo-gradients belong to.
            pseudograds: Pseudo-gradients for the fragment's parameters only.

        Returns:
            Updated global parameters for the fragment.
        """
        header = json.dumps(
            {
                "worker_id": worker_id,
                "fragment_id": fragment_id,
            }
        ).encode("utf-8")
        tensor_data = self._serialize_state_dict(pseudograds)

        body = struct.pack("!I", len(header)) + header + tensor_data

        t0 = time.time()
        params = self._request_tensor("POST", "/submit_fragment_pseudograd", body=body)
        elapsed = time.time() - t0

        logger.debug(
            f"Fragment {fragment_id} sync for {worker_id}: "
            f"sent {len(tensor_data) / 1e6:.1f} MB, "
            f"took {elapsed:.1f}s"
        )

        return params

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Fetch current global parameters (for late joiners or recovery)."""
        return self._request_tensor("GET", "/global_params")

    def heartbeat(
        self,
        worker_id: str,
        steps_per_second: float = 0.0,
        stats: Optional[dict] = None,
    ) -> dict:
        """
        Send heartbeat to server.

        Args:
            worker_id: Worker identifier.
            steps_per_second: Current training speed.
            stats: Optional unified-stats snapshot (normalized schema, see
                ``diloco/stats.py``) the server folds into its aggregate view.
                Omitted from the request when ``None``.

        Returns:
            Server status dict with sync_round, num_workers, etc.
        """
        body = {
            "worker_id": worker_id,
            "steps_per_second": steps_per_second,
        }
        if stats:
            body["stats"] = stats
        return self._request_json("POST", "/heartbeat", body)

    def deregister(self, worker_id: str):
        """Deregister from the server."""
        try:
            self._request_json(
                "POST", "/deregister", {"worker_id": worker_id}, retries=1
            )
            logger.info(f"Deregistered {worker_id} from server")
        except Exception as e:
            logger.warning(f"Failed to deregister {worker_id}: {e}")

    def get_status(self) -> dict:
        """Get server status."""
        return self._request_json("GET", "/status")

    def get_stats_history(self, max_points: int = 2000) -> dict:
        """Get the server's aggregate-stats history (for plotting).

        Returns ``{"records": [...], "count": int, "downsampled": bool}``;
        ``records`` are the logged aggregate snapshots, downsampled to at most
        ``max_points`` (latest always kept). Empty when the server has no
        output_dir / nothing logged yet.
        """
        return self._request_json("GET", f"/stats_history?max_points={int(max_points)}")

    def relay_command(self, command: str, worker_id: Optional[str] = None) -> dict:
        """Queue a trainer-control command for one or all workers.

        Posts to ``/control/command``; the server stores it and delivers it
        on the target worker(s)' next heartbeat, which applies it to the
        trainer loop. ``command`` is one of ``"save_checkpoint"``,
        ``"save_and_stop"``, ``"abort"``. ``worker_id=None`` broadcasts to
        every currently-registered worker.

        Returns the server's ``{"status", "command", "workers"}`` ack.
        """
        body: Dict[str, Any] = {"command": command}
        if worker_id is not None:
            body["worker_id"] = worker_id
        return self._request_json("POST", "/control/command", body)

    def save_state(self) -> dict:
        """Ask the server to checkpoint its state to disk now."""
        return self._request_json("POST", "/control/save_state", {})

    def shutdown(self) -> dict:
        """Ask the server to stop. The server replies then exits, so a
        connection-reset after the ack is expected and not an error."""
        return self._request_json("POST", "/control/shutdown", {}, retries=1)

    def get_known_workers(self) -> dict:
        """Get the roster of every worker_id the server has ever seen.

        Returns ``{"workers": [{worker_id, output_dir, last_registered,
        running}, ...]}``. The webui offers the not-running entries so an
        operator can relaunch a worker under its old id and resume from
        that worker's checkpoint (issue #103). Persisted with the server's
        checkpoints, so it survives a server restart.
        """
        return self._request_json("GET", "/known_workers")

    def get_info(self) -> dict:
        """Get the server's static-ish configuration.

        Returns the negotiation facts a worker needs before registering:
        the authoritative ``expected_client_settings`` (sync_every,
        bf16_comm, dylu, num_fragments_*, heartbeat_timeout), the model
        fingerprint (``model_hash``), and topology info. Distinct from
        ``get_status`` which is the live, rapidly-changing snapshot.
        """
        return self._request_json("GET", "/info")

    def fetch_model_def(self, dest_dir: str) -> str:
        """Fetch the model-definition bundle and extract it into ``dest_dir``.

        GETs ``/model_def`` (control-plane, bearer-authenticated), validates
        the ``X-Forgather-Model-Hash`` header against the server's
        ``/info`` ``model_hash`` so the definition is never silently paired
        with a mismatched parameter set, and extracts the tar traversal-safe
        (see ``model_def.extract_model_def``). Weights are never in the
        bundle — they arrive via the parameter sync.

        Returns the server's model hash (the validated bundle identity), so
        the caller can stamp the staging dir for later cache reuse.
        """
        from forgather.ml.diloco.model_def import MODEL_HASH_HEADER, extract_model_def

        url = self._url("/model_def")
        req = urllib.request.Request(url, method="GET", headers=self._headers())
        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout, context=self._ssl_for_request(url)
            ) as resp:
                bundle_hash = resp.headers.get(MODEL_HASH_HEADER)
                data = resp.read()
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8", errors="replace")
                error_detail = json.loads(error_body).get("error", error_body)
            except Exception:
                error_detail = str(e)
            raise ConnectionError(
                f"Server returned HTTP {e.code} for {url}: {error_detail}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to fetch model definition from {url}: {e}"
                f"{self._scheme_hint()}"
            ) from e

        # Cross-check the bundle against the advertised model_hash. A
        # mismatch means the server's definition and parameter set have
        # drifted (e.g. a restart on a different model mid-fetch) — fail
        # loud rather than build the wrong model.
        info_hash = self.get_info().get("model_hash")
        if bundle_hash and info_hash and bundle_hash != info_hash:
            raise DiLoCoModelMismatchError(
                f"model_def bundle hash {bundle_hash[:12]} does not match "
                f"server /info model_hash {info_hash[:12]}",
                diagnostic=(
                    "The server's model definition and parameter set "
                    "disagree; it may have been restarted on a different "
                    "model. Retry once the server has settled."
                ),
            )
        extract_model_def(data, dest_dir)
        return bundle_hash or info_hash or ""

    # ------------------------------------------------------------------
    # Work-unit dispatch (see docs/design/diloco-work-unit-dispatch.md)
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        worker_id: str,
        dataset_id: str,
        shuffle_seed: int,
        hint: dict,
    ) -> dict:
        """Register (or confirm) a ``(dataset_id, shuffle_seed)`` work queue.

        Returns ``{"total_units": K}`` — the configured per-queue
        K from the server. Subsequent registrations of the same
        (dataset_id, shuffle_seed) return the same value.

        Raises on 409 (length mismatch against a prior registration of
        the same dataset_id) — the worker should treat this as a fatal
        config error and exit.
        """
        return self._request_json(
            "POST",
            "/datasets/register",
            {
                "worker_id": worker_id,
                "dataset_id": dataset_id,
                "shuffle_seed": int(shuffle_seed),
                "hint": hint,
            },
        )

    def request_work(self, worker_id: str, dataset_id: str, shuffle_seed: int) -> dict:
        """Ask the server for the next available work unit.

        Returns ``{"unit_id": int}`` or ``{"exhausted": true}`` when
        the queue is drained.
        """
        return self._request_json(
            "POST",
            "/work/request",
            {
                "worker_id": worker_id,
                "dataset_id": dataset_id,
                "shuffle_seed": int(shuffle_seed),
            },
        )

    def complete_work(
        self,
        worker_id: str,
        dataset_id: str,
        shuffle_seed: int,
        unit_id: int,
    ) -> dict:
        """Mark a unit as confirmed-completed (diagnostic only).

        Idempotent. Workers can skip this — issuance is one-way and
        nothing about the queue's correctness path depends on
        completion acks.
        """
        return self._request_json(
            "POST",
            "/work/complete",
            {
                "worker_id": worker_id,
                "dataset_id": dataset_id,
                "shuffle_seed": int(shuffle_seed),
                "unit_id": int(unit_id),
            },
        )

    def get_work_queues(self) -> list:
        """List all active work queues (summaries only, no bitmaps)."""
        return self._request_json("GET", "/work/queues")

    def get_work_queue(self, dataset_id: str, shuffle_seed: int) -> dict:
        """Get full state of a single queue including base64 bitmaps."""
        from urllib.parse import quote

        path = (
            f"/work/queue?dataset_id={quote(dataset_id, safe='')}"
            f"&shuffle_seed={int(shuffle_seed)}"
        )
        return self._request_json("GET", path)
