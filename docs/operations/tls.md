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

> **Don't run `forgather tls init` on host B.** Init creates a *new*
> CA. Host A won't trust certs minted by host B's CA, and vice versa
> — peer-pull will fail closed and you'll get a confusing "fetch
> failed" in the Nodes view. Always use `mint` on the CA holder and
> `install` on the peer.

### On the CA-holding host (host A)

```bash
# Provision the CA and a server cert for host A.
forgather tls init --hostname a.lan --hostname b.lan \
                   --ip 10.0.0.5 --ip 10.0.0.6

# Mint a server cert for host B (writes server.crt, server.key, ca.crt
# into /tmp/b-tls/ — key is mode 0600 from creation).
forgather tls mint --hostname b.lan --ip 10.0.0.6 -o /tmp/b-tls
```

### Distribute to host B

Use a channel that preserves the 0600 mode on `server.key`:

```bash
# scp preserves permissions when copying file-by-file.
scp /tmp/b-tls/server.crt /tmp/b-tls/server.key /tmp/b-tls/ca.crt \
    b.lan:/tmp/b-tls/
# Verify mode after transfer.
ssh b.lan 'ls -l /tmp/b-tls/server.key'   # expect -rw-------
```

Email, Slack DMs, public S3 buckets — anywhere `server.key` could be
read by an unauthorized party — are off-limits. `ca.crt` is safe to
distribute over any channel (it carries no secret), but see the
warning in "Trusting the CA" below: anyone who *trusts* it can be
deceived by certs signed by it.

### On host B

```bash
forgather tls install --cert /tmp/b-tls/server.crt \
                      --key  /tmp/b-tls/server.key \
                      --ca   /tmp/b-tls/ca.crt
forgather tls status
```

`install` cross-validates that the cert's public key matches the
supplied private key, that the cert chains to the supplied CA, and
that the CA cert is actually a CA. It then writes the key with
mode 0600 from creation (no TOCTOU window), imports the CA into the
trust bundle, populates the SAN list from the cert, and sets
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

Leaf certs expire after 825 days; the CA after ten years. `forgather
tls status` warns when the server cert is within 30 days of expiry.

```bash
# Server cert only — most common. Extend SANs while you're here if
# the host's hostname/IP changed since init.
forgather tls renew --server --add-hostname new.lan --add-ip 10.0.0.7

# Re-issue the CA too. DESTRUCTIVE: every peer's trust bundle breaks
# until you redistribute the new ca.crt. Prompts for confirmation.
forgather tls renew --ca
```

After renewing a leaf cert, restart the servers that loaded the old
one. After renewing the CA:

1. `forgather tls export-ca -o /tmp/new-ca.crt` on the CA holder.
2. scp the file to every peer.
3. On each peer, `forgather tls install --ca /tmp/new-ca.crt` (or
   manually replace `~/.config/forgather/tls/ca/ca.crt` and run
   `forgather tls status` to rebuild the bundle).
4. Restart servers on every peer.

If you're stuck because half the cluster has the old CA and half the
new, redistribute the new CA to the stragglers and restart them; the
peer-pull will recover within one tick.

## Verifying the deployment

```bash
# 1. CA + server cert state.
forgather tls status

# 2. Direct OpenSSL probe — confirms the cert is what you expect.
openssl s_client -connect 127.0.0.1:8765 \
    -CAfile ~/.config/forgather/tls/ca-bundle.crt </dev/null 2>&1 \
    | grep -E "subject|issuer|Verify return"

# 3. curl over the CA bundle.
curl --cacert ~/.config/forgather/tls/ca/ca.crt \
    https://$(hostname):8765/api/health

# 4. From a peer (after `tls install` / `import-ca`).
forgather sched status   # uses the shared bundle automatically
```

## Trusting the CA from a browser

`forgather tls trust-system` prints OS-specific instructions for
installing the CA into the system trust store. Per-browser caveats:

* Firefox uses its own store: `about:preferences#privacy → View
  Certificates → Authorities → Import`.
* Chromium/Edge/Safari follow the system store.

For one-off testing, your browser's "advanced → proceed anyway"
flow still works — but cluster peer-pull requires the CA in the
shared bundle, so distribute it properly.

> **`ca.crt` is a high-trust artifact.** A machine that trusts this
> CA will accept *any* cert signed by it for *any* hostname. If a
> colleague's laptop trusts your CA, an attacker who steals your
> CA private key (`ca/ca.key`) can mint a cert claiming to be
> `bank.example.com` and that laptop will accept it without warning.
> Only trust the CA on machines you intend to talk to forgather
> servers from, and treat `ca/ca.key` with the same care as an SSH
> private key (0600, never copied, never on shared storage).

## Behind a reverse proxy or in containers

If you front your forgather servers with nginx/Caddy/Traefik that
terminates TLS, run forgather itself with `--no-tls --insecure` and
let the proxy handle the cert. Same pattern in a sidecar-style
Docker setup (TLS-terminating proxy container forwards to the
plaintext forgather container on a private network).

The runtime image's `NO_AUTH=1` smoke-test mode is for trusted-LAN
testing only. For multi-node clusters in production, either
provision TLS on the host before launching the container (mount
`~/.config/forgather/tls/` into the container) or place a TLS
terminator in front and use `--insecure` inside.

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

For corporate PKI or [mkcert](https://github.com/FiloSottile/mkcert)
workflows that already issue per-host certs, skip `forgather tls
init` and pass the cert/key directly:

```bash
# Corporate PKI:
forgather server --tls --tls-cert /etc/ssl/forgather.crt \
                       --tls-key  /etc/ssl/forgather.key

# mkcert (produces <host>.pem and <host>-key.pem):
mkcert myhost
forgather server --tls --tls-cert myhost.pem --tls-key myhost-key.pem
```

The shared config still controls the CLI client's default scheme +
trust bundle, so you can mix BYOC servers with `forgather tls
import-ca <your-corporate-ca.crt>` on the client side.

## Disabling TLS

Three options, by increasing scope:

```bash
# Per-invocation override (keeps everything on disk).
forgather server -H 127.0.0.1 --no-tls

# All servers on this host (keeps certs on disk; reversible with `tls enable`).
forgather tls disable

# Nuke everything (irreversible — you'll have to re-init).
forgather tls clean --yes
```

`tls disable` is the right tool for "I'm troubleshooting and want
HTTP back for an hour." `tls clean` is the right tool for "I'm
done with this machine."

Non-loopback HTTP still requires `--insecure` to acknowledge the
cleartext-bearer-token risk regardless of which option you picked.

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
  channel. For paranoid setups, wrap forgather in nginx with
  `ssl_verify_client on` and require client certs at the proxy.
* **Bearer tokens in URLs.** The webui's startup banner and the
  WebSocket TTY-stream URL carry the token as a query parameter
  (`?token=…`). TLS protects the wire, but the token can still
  appear in browser history and uvicorn access logs. Treat the
  banner URL like a password; copy-paste it once into a real
  bookmark rather than leaving it in shell history.
* **Cluster peer trust.** With TLS, the master peer-pulls peers
  over HTTPS validating against the shared CA bundle. The
  inter-node carve-out in `auth.py` still uses source-IP matching
  to identify peers — meaning an attacker who steals a peer's
  cert+key *and* lands on an IP in the cluster's member table can
  exercise the narrow mutation endpoints (`training_local`,
  `gpu_policy_local`). Strong cert-to-identity binding is a v2
  item; for now, the operational mitigation is the same as it
  was pre-TLS: keep cluster traffic on a trusted LAN, and tighten
  filesystem perms on `server.key` on every peer.
