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

        base = (
            base_url
            or os.environ.get("FORGATHER_SERVER_URL")
            or _default_base_url()
        )
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
        if self.base.lower().startswith("https://"):
            try:
                from forgather.tls import load_config
                from forgather.tls.runtime import httpx_verify  # noqa: F401

                cfg = load_config()
                bundle = cfg.effective_bundle()
                if bundle is not None:
                    self.session.verify = str(bundle)
                if not cfg.verify_hostname:
                    self.session.mount(
                        "https://", _NoHostnameHTTPSAdapter.build()
                    )
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

    def job_delete(self, job_id):
        return self._delete(f"/jobs/{job_id}").json()

    def cleanup_jobs(self):
        return self._post("/jobs/cleanup").json()

    def gc_jobs(self):
        return self._post("/jobs/gc").json()

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
        try:
            ws = await websockets.connect(ws_url)
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
    ):
        """Fan out a multi-node training submit.

        ``members`` is a list of dicts with keys ``node_id``,
        ``nproc_per_node``, and optional ``nccl_socket_ifname``. The
        server matches each member to a known cluster peer, derives
        the iface from the member's advertised IP when omitted, and
        spawns torchrun with the right rdzv args on every peer.
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
        return self._post("/cluster/jobs/submit", body).json()

    def cluster_job_cancel(self, cluster_job_id):
        """Fan out cancel to every participant of the bundle."""
        return self._post(f"/cluster/jobs/{cluster_job_id}/cancel").json()
