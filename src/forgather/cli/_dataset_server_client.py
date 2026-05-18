"""Tiny HTTP client used by the `forgather dataset-server` diagnostic
subcommands.

Uses :mod:`urllib` (stdlib) so the CLI doesn't pull in extra deps.
"""

from __future__ import annotations

import json
import os
import ssl
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .dataset_server_args import DEFAULT_SERVER_URL, SERVER_URL_ENV


def _ssl_context_for_url(
    url: str, *, insecure: bool = False
) -> Optional[ssl.SSLContext]:
    """Build an SSLContext that trusts our local CA bundle (if any).

    Returns ``None`` for plain ``http://`` URLs. Honors the shared
    config's ``verify_hostname`` flag — defaults to chain-only
    validation, matching the rest of the forgather TLS surface.

    ``insecure=True`` returns an SSL context that *neither* verifies
    chains *nor* checks hostnames. Used for the explicit-opt-out
    case (``--insecure``) where the operator knows the channel is
    secured by some other means (SSH tunnel, VPN, air-gapped LAN).
    """
    if not url.lower().startswith("https://"):
        return None
    if insecure:
        # Operator-asserted "I trust this channel for other reasons"
        # — chain-and-hostname validation off. Documented on the
        # CLI flag; not the default.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        from forgather.tls import httpx_verify

        verify = httpx_verify()
    except Exception:
        verify = True
    if isinstance(verify, ssl.SSLContext):
        return verify
    if isinstance(verify, str):
        return ssl.create_default_context(cafile=verify)
    return ssl.create_default_context()


class ServerError(RuntimeError):
    pass


class AuthRequired(ServerError):
    pass


def resolve_server_url(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env = os.environ.get(SERVER_URL_ENV)
    if env:
        return env
    # Default scheme follows local TLS state — when the operator ran
    # `forgather tls init` and the local server is serving HTTPS, the
    # CLI talks HTTPS automatically against 127.0.0.1.
    try:
        from forgather.tls import client_scheme

        scheme = client_scheme()
    except Exception:
        scheme = "http"
    if scheme == "https":
        # Replace the default URL's scheme.
        return DEFAULT_SERVER_URL.replace("http://", "https://", 1)
    return DEFAULT_SERVER_URL


def resolve_token(url: str, explicit: Optional[str]) -> Optional[str]:
    """Bearer token discovery: explicit > env var > localhost token file."""
    if explicit:
        return explicit
    env = os.environ.get("FORGATHER_DATASET_SERVER_TOKEN")
    if env:
        return env.strip() or None
    # Reuse the same helper RemoteBackend uses so behaviour matches.
    try:
        from forgather.ml.datasets.remote_backend import resolve_auth_token
    except ImportError:
        return None
    return resolve_auth_token(url, explicit=None)


def _build_headers(token: Optional[str]) -> Dict[str, str]:
    h: Dict[str, str] = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _request(
    method: str,
    url: str,
    token: Optional[str] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    insecure: bool = False,
) -> Any:
    data = None
    headers = _build_headers(token)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, method=method, headers=headers)
    ctx = _ssl_context_for_url(url, insecure=insecure)
    try:
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            err = json.loads(exc.read().decode("utf-8")).get("error") or exc.reason
        except Exception:
            err = exc.reason
        if exc.code == 401:
            raise AuthRequired(
                f"401 from {url}: {err} — check --token or "
                "$FORGATHER_DATASET_SERVER_TOKEN"
            ) from exc
        raise ServerError(f"{exc.code} from {url}: {err}") from exc
    except URLError as exc:
        raise ServerError(
            f"could not reach dataset server at {url}: {exc.reason}"
        ) from exc


class DatasetServerClient:
    """Minimal client for the dataset server's diagnostic endpoints."""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        insecure: bool = False,
    ):
        self.url = resolve_server_url(url).rstrip("/")
        self.token = resolve_token(self.url, token)
        # When True, every outbound request skips TLS chain + hostname
        # validation. Operator-asserted "I trust this channel"; used
        # for SSH-tunneled and otherwise out-of-band-secured remotes.
        self.insecure = insecure

    @classmethod
    def from_args(cls, args) -> "DatasetServerClient":
        return cls(
            url=getattr(args, "server", None),
            token=getattr(args, "token", None),
            insecure=bool(getattr(args, "insecure", False)),
        )

    # Endpoints (one method per /v1 path the diagnostic actions hit).

    def health(self) -> Dict[str, Any]:
        return _request(
            "GET", f"{self.url}/v1/health", token=self.token, insecure=self.insecure
        )

    def auth_status(self) -> Dict[str, Any]:
        return _request(
            "GET",
            f"{self.url}/v1/auth/status",
            token=self.token,
            insecure=self.insecure,
        )

    def list_datasets(self) -> Dict[str, Any]:
        return _request(
            "GET", f"{self.url}/v1/datasets", token=self.token, insecure=self.insecure
        )

    def list_local(self) -> Dict[str, Any]:
        return _request(
            "GET", f"{self.url}/v1/local", token=self.token, insecure=self.insecure
        )

    def list_hf_cache(self) -> Dict[str, Any]:
        return _request(
            "GET", f"{self.url}/v1/cache/hf", token=self.token, insecure=self.insecure
        )
