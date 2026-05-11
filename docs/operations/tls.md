# TLS for Forgather servers

Forgather ships three FastAPI/Uvicorn servers — `forgather server`,
`dataset_server`, and `inference_server`. All three speak HTTPS off
the same per-host config, so configuring TLS once enables it
everywhere.

This page walks through the single-host and multi-node setups, plus
renewal and trust distribution. See `forgather tls --help` for the
full subcommand reference.

## Where state lives

A single directory holds the CA, server cert, trust bundle, and
config:

```
~/.config/forgather/tls/
├── config.yaml          # single source of truth
├── ca/
│   ├── ca.crt           # local CA (distribute to peers/clients)
│   ├── ca.key           # 0600; only on CA-holding hosts
│   └── ca.srl           # serial counter
├── server.crt           # this host's server cert
├── server.key           # 0600
├── trusted/<name>.crt   # CA certs imported from other hosts
└── ca-bundle.crt        # ca.crt + every trusted/*.crt (auto-built)
```

Override the root via `$FORGATHER_TLS_DIR` (useful for tests or
multi-tenant setups).

## Single host

```bash
forgather tls init
forgather tls status
forgather server -H 0.0.0.0          # auto-on: HTTPS, refuses to bind without TLS
forgather dataset-server start -H 0.0.0.0
forgather inf server -H 0.0.0.0 -m output_models/my_model
```

`forgather tls init` auto-detects hostnames (`socket.gethostname()`,
`socket.getfqdn()`) and LAN IPs (psutil), then mints a server cert
whose Subject Alternative Names cover all of them. Pass extras with
`--hostname` / `--ip` if discovery missed an alias.

After init, every server respects:

* **Loopback bind** (`127.0.0.1`, `::1`, `localhost`): TLS still kicks
  in when `enabled: true` is in the shared config. Pass `--no-tls`
  to keep loopback in cleartext.
* **Non-loopback bind**: refused unless TLS is provisioned, or
  `--insecure` is passed (cleartext bearer tokens — only suitable for
  an SSH-tunneled or VPN-only LAN).
* **`--tls` / `--no-tls` flags**: per-invocation override.
* **`--tls-cert PATH` / `--tls-key PATH` flags**: bring-your-own
  cert/key (escape hatch for corporate PKI). Skips the shared-CA
  path.

## Multi-node cluster

Run the same CA across every node so peer-pull validates without
warnings.

### On the CA-holding host (host A)

```bash
# Provision the CA and a server cert for host A.
forgather tls init --hostname a.lan --hostname b.lan \
                   --ip 10.0.0.5 --ip 10.0.0.6

# Mint a server cert for host B.
forgather tls mint --hostname b.lan --ip 10.0.0.6 -o /tmp/b-tls

# Distribute the CA cert (no key) — also embedded in /tmp/b-tls/ca.crt.
forgather tls export-ca -o /tmp/forgather-ca.crt
```

Copy `/tmp/b-tls/` (server.crt, server.key, ca.crt) to host B via scp,
NFS, or any other channel that protects the private key.

### On host B

```bash
forgather tls install --cert /tmp/b-tls/server.crt \
                      --key  /tmp/b-tls/server.key \
                      --ca   /tmp/b-tls/ca.crt
forgather tls status
```

`install` writes the cert/key into the shared TLS dir, imports the CA
into the trust bundle, populates the SAN list from the cert, and sets
`enabled: true`. Host B can serve TLS but cannot mint new certs (no
CA private key).

### Start both servers

```bash
# Host A and host B
forgather server -H 0.0.0.0 --cluster mycluster
```

mDNS advertisements include a `tls=1` TXT record so peers know which
scheme to use. The peer-pull loop dials `https://...` and uses the
shared CA bundle to validate.

## Renewal

Leaf certs expire after 825 days; the CA after ten years.

```bash
# Server cert only (most common — extend SANs while you're here).
forgather tls renew --server --add-hostname new.lan

# Re-issue the CA too. DESTRUCTIVE: every peer's trust bundle breaks
# until you redistribute the new ca.crt.
forgather tls renew --ca
```

After renewing a leaf cert, restart the servers that loaded the old
one. After renewing the CA, redistribute it to every peer's
`tls/ca/ca.crt`, then re-run `forgather tls install` on the peers.

## Trusting the CA from a browser

`forgather tls trust-system` prints OS-specific instructions for
installing the CA into the system trust store. Per-browser caveats:

* Firefox uses its own store: `about:preferences#privacy → View
  Certificates → Authorities → Import`.
* Chromium/Edge/Safari follow the system store.

For one-off testing, your browser's "advanced → proceed anyway"
flow still works — but cluster peer-pull requires the CA in the
shared bundle, so distribute it properly.

## CLI clients

CLI clients (`forgather control`, `forgather job`, `forgather sched`,
`forgather gpu`, `forgather cluster`, `forgather dataset-server
status|list|cache|local`) pick the scheme + CA bundle up from the
shared config automatically. Override with the env vars:

```bash
export FORGATHER_SERVER_URL=https://my-server.lan:8765
export FORGATHER_DATASET_SERVER=https://my-dataset.lan:8766
```

If the URL is `https://`, the client uses
`~/.config/forgather/tls/ca-bundle.crt` as the trust anchor.

## Bring-your-own certs

For corporate PKI or mkcert workflows that already issue per-host
certs, skip `forgather tls init` and pass the cert/key directly:

```bash
forgather server --tls --tls-cert /etc/ssl/forgather.crt \
                       --tls-key  /etc/ssl/forgather.key
```

The shared config still controls the CLI client's default scheme +
trust bundle, so you can mix BYOC servers with `forgather tls
import-ca <your-corporate-ca.crt>` on the client side.

## Disabling TLS

```bash
forgather tls clean --yes           # nukes ~/.config/forgather/tls/
```

Or per-invocation:

```bash
forgather server -H 127.0.0.1 --no-tls
```

Non-loopback HTTP still requires `--insecure` to acknowledge the
cleartext-bearer-token risk.

## Threat model

* The CA private key never leaves the host that minted it. Only the
  CA cert is distributed.
* Leaf certs cover SAN entries that the server itself binds. A peer
  that advertised `10.0.0.6` but presents a cert without that SAN
  will fail strict-TLS verification on the dial-out path.
* Bearer-token auth is unchanged — TLS just protects the token (and
  request/response bodies) in transit. Setting `--no-auth` is still
  needed for unauthenticated access.
* mTLS (client certs) is not implemented in v1. Clients authenticate
  via the existing bearer-token mechanism over a TLS-protected
  channel.
