"""TLS verification, certificate import, and model listing for agent profiles.

The user's vLLM box runs TLS with a self-signed cert on a LAN, so the
system CA check fails. Two supported postures per profile, mirroring how
the inference registry handles the same problem:

- **verify_tls = False** — accept any cert (no chain/hostname check). One
  click, but vulnerable to a MITM on the path. Fine for a trusted LAN.
- **import the cert** — fetch the server's certificate (TOFU), store its
  PEM in the profile (``ca_cert_pem``), and verify against it with hostname
  checking off (same posture as forgather's own private-CA peers: the
  chain check is the real boundary; SANs on LAN IPs are theatre).

``build_verify`` turns a profile into the value httpx wants for ``verify=``.
``fetch_server_cert`` retrieves a PEM for the import flow.
``list_models`` queries the model list (Claude SDK, or vLLM/OpenAI
``/v1/models``) so the webui can offer a picker instead of a typed-in name.
"""

from __future__ import annotations

import logging
import socket
import ssl
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger("forgather_server.agent_tls")

_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)


def build_verify(*, base_url: str, verify_tls: bool, ca_cert_pem: str) -> Any:
    """Return the value to pass as httpx ``verify=`` for ``base_url``.

    - plain http -> True (no-op; httpx ignores it)
    - verify_tls False -> False (accept any cert)
    - ca_cert_pem set -> an SSLContext trusting that cert, hostname check off
    - else -> True (system trust store)
    """
    if not (base_url or "").lower().startswith("https"):
        return True
    if not verify_tls:
        return False
    if ca_cert_pem and ca_cert_pem.strip():
        ctx = ssl.create_default_context(cadata=ca_cert_pem)
        # LAN self-signed certs rarely carry a SAN matching the IP/host the
        # operator dials; the chain check against the imported cert is the
        # real boundary. Matches forgather.tls.httpx_verify's posture.
        ctx.check_hostname = False
        return ctx
    return True


def _host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"unsupported scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("missing host in base_url")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def fetch_server_cert(base_url: str) -> dict:
    """Fetch the server's leaf certificate (PEM + SHA-256 fingerprint).

    Trust-on-first-use: connects without verifying, grabs the presented
    cert, and returns it for the operator to review and import. Does NOT
    persist anything — the caller saves the PEM into a profile if accepted.
    """
    host, port = _host_port(base_url)
    # Unverified context purely to retrieve the cert bytes for review.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=10) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            der = ssock.getpeercert(binary_form=True)
    if not der:
        raise RuntimeError("server presented no certificate")
    pem = ssl.DER_cert_to_PEM_cert(der)
    import hashlib

    fp = hashlib.sha256(der).hexdigest()
    fingerprint = ":".join(fp[i : i + 2] for i in range(0, len(fp), 2)).upper()
    return {"host": host, "port": port, "pem": pem, "sha256": fingerprint}


def list_models(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    verify_tls: bool,
    ca_cert_pem: str,
) -> List[Dict[str, Any]]:
    """Return available models for a connection.

    Each entry is ``{"id": str, "max_model_len": Optional[int]}``.
    ``max_model_len`` is the server-reported context window (vLLM includes
    it in the model card) and is ``None`` when the provider doesn't report
    it (e.g. Claude via the SDK). The agent uses it to size ``max_tokens``
    so the operator doesn't have to look the context up per model.

    - Claude (anthropic provider, no base_url): the Anthropic SDK's
      ``models.list()`` (no context length available → ``None``).
    - Any base_url (vLLM / OpenAI-compatible): GET ``<base_url>/v1/models``
      with a bearer; each card carries ``id`` and ``max_model_len``.
    """
    base_url = (base_url or "").rstrip("/")
    if provider == "anthropic" and not base_url:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError("the 'anthropic' package is required") from e
        client = Anthropic(api_key=api_key or "placeholder")
        return [{"id": m.id, "max_model_len": None} for m in client.models.list().data]

    # OpenAI-compatible /v1/models (vLLM and friends).
    verify = build_verify(base_url=base_url, verify_tls=verify_tls, ca_cert_pem=ca_cert_pem)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url + "/v1/models"
    with httpx.Client(timeout=_TIMEOUT, verify=verify) as client:
        r = client.get(url, headers=headers or None)
        r.raise_for_status()
        data = r.json()
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict) and it.get("id"):
            ctx = it.get("max_model_len")
            out.append(
                {
                    "id": str(it["id"]),
                    "max_model_len": int(ctx) if isinstance(ctx, int) else None,
                }
            )
    return out
