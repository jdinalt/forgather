"""HTTP/WebSocket client for the forgather-server API."""

import json
import os
from pathlib import Path
from urllib.parse import quote

from forgather.preprocess import forgather_config_dir


class ServerUnreachable(Exception):
    pass


class AuthRequired(RuntimeError):
    """Raised when the server returns 401.

    Distinct from ``ServerUnreachable`` because the fix is different:
    the server is up, the client just doesn't have a valid token.
    Inherits from ``RuntimeError`` so the CLI's existing
    ``except RuntimeError`` blocks already surface the message.
    """

    pass


def _load_auth_token():
    """Find the bearer token shared with the server.

    Order: ``$FORGATHER_SERVER_TOKEN`` overrides everything (handy for
    multi-server setups), otherwise
    ``<forgather_config_dir>/server/auth_token``. Returns ``None`` if
    neither is available — the client still issues requests, and the
    server's 401 response surfaces a clear error.
    """
    env = os.environ.get("FORGATHER_SERVER_TOKEN")
    if env:
        return env.strip()
    token_path = Path(forgather_config_dir()) / "server" / "auth_token"
    try:
        text = token_path.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None
    return text or None


def _default_base_url():
    """Same default as before, but pick ``https://`` when local TLS is on.

    This mirrors how the server itself emits its banner URL. A user who
    ran ``forgather tls init`` doesn't have to set $FORGATHER_SERVER_URL
    to talk to their own server — the client picks the scheme up from
    the shared config.
    """
    try:
        from forgather.tls import client_scheme

        scheme = client_scheme()
    except Exception:
        scheme = "http"
    return f"{scheme}://127.0.0.1:8765"


class _NoHostnameHTTPSAdapter:
    """Lazy import wrapper — only build the requests/urllib3 plumbing
    when we actually need it. Avoids importing requests at module load."""

    @staticmethod
    def build():
        from requests.adapters import HTTPAdapter

        class _Adapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                # ``assert_hostname=False`` tells urllib3 to skip the
                # cert-SAN-vs-URL-hostname check while still requiring
                # chain validation against the configured CA bundle.
                # Matches what we do for httpx/urllib elsewhere — see
                # forgather.tls.runtime.httpx_verify for the rationale.
                kwargs["assert_hostname"] = False
                return super().init_poolmanager(*args, **kwargs)

        return _Adapter()


class ServerClient:
    def __init__(self, base_url=None, timeout=30.0):
        import requests

        base = base_url or os.environ.get("FORGATHER_SERVER_URL") or _default_base_url()
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "forgather-cli"
        # HTTPS configuration:
        #   1. Point `requests` at the shared CA bundle so our self-
        #      signed certs validate. Without it, requests falls back
        #      to the system trust store and rejects the cert.
        #   2. If the shared config has verify_hostname=False (the
        #      LAN default), install an HTTPAdapter that disables
        #      urllib3's hostname-SAN check. The chain check is the
        #      actual security boundary on a private CA — see
        #      docs/operations/tls.md.
        # Captured so the WebSocket path (stream_tty → wss://) can build an
        # SSL context with the same trust material the requests session uses;
        # otherwise the `wss` handshake falls back to the system trust store
        # and rejects the self-signed cluster cert ("could not reach").
        self._tls_bundle = None
        self._verify_hostname = True
        if self.base.lower().startswith("https://"):
            try:
                from forgather.tls import load_config

                cfg = load_config()
                bundle = cfg.effective_bundle()
                if bundle is not None:
                    self.session.verify = str(bundle)
                    self._tls_bundle = str(bundle)
                self._verify_hostname = cfg.verify_hostname
                if not cfg.verify_hostname:
                    self.session.mount("https://", _NoHostnameHTTPSAdapter.build())
            except Exception:
                pass
        self._token = _load_auth_token()
        if self._token:
            self.session.headers["Authorization"] = f"Bearer {self._token}"

    @classmethod
    def from_args(cls, args):
        return cls(getattr(args, "server", None))

    def _url(self, path):
        return f"{self.base}/api{path}"

    def _ws_url(self, path):
        url = self._url(path)
        if url.startswith("https://"):
            return "wss://" + url[len("https://") :]
        return "ws://" + url[len("http://") :]

    def _check_response(self, r):
        if r.status_code == 401:
            # A 401 carrying this tag came from an *upstream* server the
            # forgather server was proxying to (a DiLoCo/dataset server whose
            # stored token is stale), not from the forgather server itself.
            # The fix is different — re-register the upstream token — so don't
            # point the user at $FORGATHER_SERVER_TOKEN.
            if r.headers.get("x-upstream-auth-failed"):
                raise AuthRequired(
                    "the forgather server's stored token for the upstream "
                    "server was rejected. Re-register it with a fresh "
                    "--auth-token (forgather diloco register …), or check the "
                    "upstream server's auth."
                )
            raise AuthRequired(self._auth_error_message())
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"server: {detail}")
        return r

    def _auth_error_message(self):
        if self._token:
            return (
                f"forgather-server at {self.base} rejected the auth token. "
                "If the server was restarted with --regen-token, re-read "
                "~/.config/forgather/server/auth_token; otherwise check "
                "$FORGATHER_SERVER_TOKEN."
            )
        return (
            f"forgather-server at {self.base} requires authentication. "
            "Start the server (it persists a token at "
            "~/.config/forgather/server/auth_token) or set "
            "$FORGATHER_SERVER_TOKEN."
        )

    def _get(self, path, **kwargs):
        import requests

        try:
            r = self.session.get(self._url(path), timeout=self.timeout, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        return self._check_response(r)

    def _post(self, path, body=None):
        import requests

        try:
            r = self.session.post(self._url(path), json=body, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        return self._check_response(r)

    def _delete(self, path):
        import requests

        try:
            r = self.session.delete(self._url(path), timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        return self._check_response(r)

    def ping(self):
        """Cheap reachability probe for auto-detect.

        Hits the unauthenticated ``/api/health`` endpoint with a short
        timeout and returns ``True`` iff the orchestrator answers. Never
        raises — callers use it to decide between the orchestrator path
        and the direct-to-param-server path.
        """
        import requests

        try:
            r = self.session.get(self._url("/health"), timeout=min(self.timeout, 2.0))
            return r.ok
        except Exception:
            return False

    # Queue

    def enqueue_job(
        self,
        *,
        project_dir,
        config,
        job_type,
        job_params,
        requested_gpus=0,
        priority=0,
        dynamic_args=None,
        dataset_source=None,
    ):
        body = {
            "project_dir": project_dir,
            "config": config,
            "dynamic_args": dynamic_args or {},
            "requested_gpus": requested_gpus,
            "priority": priority,
            "job_type": job_type,
            "job_params": job_params,
        }
        # Only sent when set — the server treats a missing dataset_source
        # as "local" (the EnqueueRequest default). Shapes:
        #   {"kind": "auto"} | {"kind": "local"} |
        #   {"kind": "server", "server_id": "local:<qid>" | "user:<id>"}
        if dataset_source is not None:
            body["dataset_source"] = dataset_source
        return self._post("/queue", body).json()

    def enqueue_training(
        self, project_dir, config, *, dynamic_args, priority, requested_gpus
    ):
        body = {
            "project_dir": project_dir,
            "config": config,
            "dynamic_args": dynamic_args,
            "requested_gpus": requested_gpus,
            "priority": priority,
            "job_type": "training",
            "job_params": {},
        }
        return self._post("/queue", body).json()

    def list_queue(self):
        return self._get("/queue").json()

    def cancel(self, queue_id):
        return self._delete(f"/queue/{queue_id}").json()

    def get_scheduler(self):
        return self._get("/queue/scheduler").json()

    def set_scheduler(self, enabled):
        return self._post("/queue/scheduler", {"enabled": enabled}).json()

    # Jobs

    def list_jobs(self, include_dead=False):
        params = {"include_dead_endpoints": "true" if include_dead else "false"}
        return self._get("/jobs", params=params).json()

    def job_status(self, job_id):
        import requests

        try:
            r = self.session.get(
                self._url(f"/jobs/{job_id}/status"), timeout=self.timeout
            )
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        if r.status_code == 401:
            raise AuthRequired(self._auth_error_message())
        return r

    def job_control(self, job_id, action):
        return self._post(f"/jobs/{job_id}/control/{action}").json()

    def job_dump(self, job_id):
        return self._get(f"/jobs/{job_id}/tty").content

    def job_tty_path(self, job_id):
        """On-disk path of the job's captured TTY (server host), resolved by
        the same get_record path as job_dump. Raises on 404 (no such job /
        no TTY yet)."""
        return self._get(f"/jobs/{job_id}/tty-path").json().get("tty_log_path")

    def job_delete(self, job_id):
        return self._delete(f"/jobs/{job_id}").json()

    def cleanup_jobs(self):
        return self._post("/jobs/cleanup").json()

    def gc_jobs(self):
        return self._post("/jobs/gc").json()

    def _ws_ssl_context(self):
        """SSL context for wss:// matching the requests session's trust:
        the shared CA bundle (so self-signed cluster certs validate), with
        the hostname-SAN check disabled when the config sets
        verify_hostname=False (the LAN default) — chain validation is the
        real boundary on a private CA."""
        import ssl

        ctx = ssl.create_default_context(cafile=self._tls_bundle)
        if not self._verify_hostname:
            ctx.check_hostname = False
        return ctx

    async def stream_tty(self, job_id, follow=True):
        import websockets
        import websockets.exceptions

        # Token rides in the query string because not every websocket
        # client surface (notably some browser shims) lets us set
        # arbitrary request headers; the AuthMiddleware accepts both.
        qs = f"?follow={'true' if follow else 'false'}"
        if self._token:
            qs += f"&token={quote(self._token)}"
        ws_url = self._ws_url(f"/jobs/{job_id}/tty") + qs
        # For wss:// give websockets the same trust material the requests
        # session uses — the shared CA bundle, hostname-SAN check disabled on
        # the LAN default. Without this the handshake hits the system trust
        # store and rejects the self-signed cluster cert (the `--follow`
        # "could not reach" bug, while non-follow job_dump over requests
        # works). ws:// needs no context.
        connect_kwargs = {}
        if ws_url.startswith("wss://"):
            connect_kwargs["ssl"] = self._ws_ssl_context()
        try:
            ws = await websockets.connect(ws_url, **connect_kwargs)
        except OSError:
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        except websockets.exceptions.InvalidStatus as e:
            # When the auth middleware rejects a WebSocket *before* the
            # upgrade completes, uvicorn surfaces the close as a 403 to
            # the HTTP client; a successful upgrade followed by an
            # auth-close uses a 4401 application close code instead. We
            # treat both as auth failures for CLI ergonomics.
            sc = getattr(e.response, "status_code", None)
            if sc in (401, 403):
                raise AuthRequired(self._auth_error_message())
            raise
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    yield ("bytes", message)
                else:
                    try:
                        frame = json.loads(message)
                        if isinstance(frame, dict) and frame.get("type") == "error":
                            yield ("error", frame.get("detail", message))
                        else:
                            yield ("bytes", message.encode())
                    except Exception:
                        yield ("bytes", message.encode())
        except websockets.exceptions.ConnectionClosed:
            return
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # GPUs

    def list_gpus(self):
        return self._get("/gpus").json()

    def gpu_policy_all(self):
        return self._get("/gpus/policy").json()

    def set_gpu_policy(self, idx, *, disabled=None, min_priority=None):
        body = {}
        if disabled is not None:
            body["disabled"] = disabled
        if min_priority is not None:
            body["min_priority"] = min_priority
        return self._post(f"/gpus/{idx}/policy", body).json()

    def kill_gpu_processes(self, idx):
        return self._post(f"/gpus/{idx}/kill", {"confirmed": True}).json()

    # Cluster (multi-node, opt-in via the server's --cluster flag)

    def cluster_self(self):
        """This node's identity in the cluster, or null if standalone."""
        return self._get("/cluster/self").json()

    def cluster_members(self):
        """Cluster name, master node_id, and the full member table."""
        return self._get("/cluster/members").json()

    def cluster_master(self):
        """Master node_id and is_self_master."""
        return self._get("/cluster/master").json()

    def cluster_jobs_list(self):
        """List multi-node bundles (newest first), with rolled-up status."""
        return self._get("/cluster/jobs").json()

    def cluster_job_get(self, cluster_job_id):
        """Get one bundle (rolled-up status fanned out from master)."""
        return self._get(f"/cluster/jobs/{cluster_job_id}").json()

    def cluster_jobs_submit(
        self,
        *,
        project_dir,
        config,
        members,
        dynamic_args=None,
        priority=0,
        rdzv_node_id=None,
        rdzv_port=None,
        allow_version_mismatch=False,
        dataset_source=None,
        diloco=None,
    ):
        """Fan out a multi-node training submit.

        ``members`` is a list of dicts with keys ``node_id``,
        ``nproc_per_node``, and optional ``nccl_socket_ifname``. The
        server matches each member to a known cluster peer, derives
        the iface from the member's advertised IP when omitted, and
        spawns torchrun with the right rdzv args on every peer.

        ``dataset_source`` mirrors the webui submit modal's
        dataset-source choice — e.g. ``{"kind": "auto"}`` for cluster
        auto-routing or ``{"kind": "server", "server_id": "..."}`` to
        pin to a specific known server.

        ``diloco`` (optional) composes the bundle with DiLoCo: every
        per-rank training job joins the named param-server as one
        logical worker group (shared base ``worker_id``). Shape:
        ``{"server_addr": str, "worker_id": str|None,
        "heartbeat_interval": float|None}``. The master resolves
        ``worker_id`` (auto-mints a memorable default when None) and
        the server's bearer token so every peer authenticates uniformly.
        """
        body = {
            "project_dir": project_dir,
            "config": config,
            "members": members,
            "dynamic_args": dynamic_args or {},
            "priority": priority,
            "allow_version_mismatch": allow_version_mismatch,
        }
        if rdzv_node_id is not None:
            body["rdzv_node_id"] = rdzv_node_id
        if rdzv_port is not None:
            body["rdzv_port"] = rdzv_port
        if dataset_source is not None:
            body["dataset_source"] = dataset_source
        if diloco is not None:
            body["diloco"] = diloco
        return self._post("/cluster/jobs/submit", body).json()

    def cluster_job_cancel(self, cluster_job_id):
        """Fan out cancel to every participant of the bundle."""
        return self._post(f"/cluster/jobs/{cluster_job_id}/cancel").json()

    def cluster_dataset_inventory(self):
        """Master-aggregated dataset-server inventory + dataset listing.

        Returns the same payload the webui Cluster + Servers tabs
        consume — useful for verifying routing readiness from the CLI
        before kicking off cluster-mode training.
        """
        return self._get("/cluster/dataset_inventory").json()

    def cluster_dataset_resolve(self, dataset_id):
        """Ask the master's router which server it would pick for ``path``.

        Returns the response body (which contains
        ``{base_url, auth_token, server_id}`` on success). Raises
        ``RuntimeError`` on 503 (cold-start) / 410 (no candidate) /
        other 4xx with the upstream detail message — same exception
        shape as the rest of ServerClient, so the CLI handler just
        prints the message.
        """
        return self._get(
            f"/cluster/dataset_router/resolve?dataset_id={quote(dataset_id, safe='')}"
        ).json()

    def cluster_server_proxy_get(self, server_id, op):
        """Cluster-proxied GET against a single dataset_server.

        ``op`` is one of ``health``, ``auth-status``, ``datasets``,
        ``cache``, ``local``. The master injects the bearer from its
        inventory; the caller only needs the cluster bearer.
        """
        path = f"/cluster/dataset_server_proxy/{server_id}/{op}"
        return self._get(path).json()

    # DiLoCo (the orchestrator proxies to upstream parameter servers,
    # resolving each upstream's bearer + TLS verification on our behalf —
    # so these only need the orchestrator's own auth, already configured).

    def list_diloco_servers(self):
        """Unified list: locally-spawned + registered (+ cluster) servers."""
        return self._get("/diloco/servers").json()

    def list_diloco_registry(self):
        """Just the user-registered external entries."""
        return self._get("/diloco/registry").json()

    def add_diloco_registry(
        self, *, base_url, label=None, auth_token=None, verify_tls=True
    ):
        body = {"base_url": base_url, "verify_tls": verify_tls}
        if label:
            body["label"] = label
        if auth_token:
            body["auth_token"] = auth_token
        return self._post("/diloco/registry", body).json()

    def delete_diloco_registry(self, entry_id):
        return self._delete(f"/diloco/registry/{entry_id}").json()

    def generate_diloco_worker_names(self, count, exclude=None):
        return self._post(
            "/diloco/generate-worker-names",
            {"count": int(count), "exclude": list(exclude or [])},
        ).json()

    def diloco_server_status(self, base):
        return self._get(f"/diloco/server-status?base={quote(base, safe='')}").json()

    def diloco_server_info(self, base):
        return self._get(f"/diloco/server-info?base={quote(base, safe='')}").json()

    def diloco_known_workers(self, base):
        return self._get(f"/diloco/known-workers?base={quote(base, safe='')}").json()

    def diloco_work_queues(self, base):
        return self._get(f"/diloco/work-queues?base={quote(base, safe='')}").json()

    def diloco_stats_history(self, base, max_points=2000):
        return self._get(
            f"/diloco/stats-history?base={quote(base, safe='')}"
            f"&max_points={int(max_points)}"
        ).json()

    def diloco_work_queue(self, base, dataset_id, shuffle_seed):
        return self._get(
            f"/diloco/work-queue?base={quote(base, safe='')}"
            f"&dataset_id={quote(str(dataset_id), safe='')}"
            f"&shuffle_seed={int(shuffle_seed)}"
        ).json()

    def diloco_server_control(self, action, base, command=None, worker_id=None):
        body = {}
        if command is not None:
            body["command"] = command
        if worker_id is not None:
            body["worker_id"] = worker_id
        return self._post(
            f"/diloco/server-control/{action}?base={quote(base, safe='')}", body
        ).json()
