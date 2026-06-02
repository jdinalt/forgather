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
forgather dataset-server start --local-only -H 0.0.0.0
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

### TL;DR — three commands

For a typical LAN cluster where you have ssh access to every peer:

```bash
# 1. On the chosen CA holder (your dev workstation is the usual pick):
forgather tls init

# 2. For each other node — one line per node, no IP table to track:
forgather tls deploy --force [--container <NAME>] <NODE>
#   <NODE> is anything ssh can resolve (hostname, IP, ~/.ssh/config alias).
#   --container <NAME>: pass when the peer runs forgather inside a Docker
#                       container (the install + token write happen via
#                       `docker exec <NAME>`, no host-side file shuffling).
#   --force:            overwrite any pre-existing TLS state on the peer.
#                       Omit if you're sure the peer is clean.

# 3. Restart `forgather server -H 0.0.0.0 --cluster <name>` on every node
#    so each one picks up its new cert.
```

That's it for the cluster-TLS bring-up. The numbered Step-1/Step-2/
Step-3 sections below break the same procedure into its component
parts (useful when you don't have ssh, or want to inspect each
cert before installing it), and the "Trust model" section explains
*why* one CA + chain-only validation is the right design for a
LAN cluster.

### Trust model: one CA, chain-only validation

A forgather cluster has **exactly one CA**. One host — call it host A
— holds the CA private key and mints leaf certs for every node,
including itself. Other hosts (B, C, …) never mint anything; they
just install the cert+key that A produced for them, plus a copy of
A's CA cert.

That single CA is the trust anchor on **every** node, so peer-pull
verifies in both directions:

* **A → B**: B presents a cert signed by A's CA. A's bundle contains
  A's CA. ✓
* **B → A**: A presents a cert signed by A's CA. B's bundle contains
  A's CA (you copied it there via `install --ca`). ✓
* **B → C** (three-node case): C presents a cert signed by A's CA.
  B's bundle contains A's CA. ✓

You do **not** need to mint a second cert "back from B to A". The
asymmetry is in *issuance* (only A has the CA key), not in *trust*
(every node trusts the same CA).

**Chain-only validation is the default.** Standard TLS layers two
checks: (1) the cert chains to a trusted CA, and (2) the cert's SAN
matches the URL hostname/IP. On the public web, (2) is what binds a
cert to a specific operator — the public CA refused to sign for a
domain you don't control. On a private LAN with a private CA, (2)
adds nothing: the operator who holds the CA key can mint a cert
claiming any hostname or IP, and the IPs themselves are often
DHCP-issued and ephemeral. So forgather's default is **chain-only**:
the peer's cert must chain to your CA; the SAN is informational.

The practical consequence: you don't need to know each peer's
hostname or IP when minting its cert. The `--hostname` / `--ip`
flags are optional in `forgather tls mint`. Pass them only when an
external client (a browser, a non-forgather tool) needs to hit the
URL by a specific name and you want that client to enforce
hostname-SAN matching.

For paranoid setups (public-DNS clusters, regulated environments
where strict RFC-6125 verification is a policy requirement), flip
`verify_hostname: true` in `~/.config/forgather/tls/config.yaml`
and make sure your mint commands carry the right `--hostname`/`--ip`
SAN entries.

### What if host A goes away?

The CA's job is **issuing** certs, not validating them. Validation
only needs the CA's *public* cert (`ca.crt`), and every peer
already has a copy in its bundle. So:

* **Temporary A outage** (reboot, network blip, crashed forgather
  server): the rest of the cluster keeps working. B ↔ C peer-pull
  continues over HTTPS, verifies fine, and the master node
  automatically rolls over (`master_node_id` is the lowest UUID
  among *reachable* members — if A had the lowest UUID, B takes
  over until A returns). You just can't mint new certs or onboard
  new peers while A is down.

* **Permanent loss of A** (disk crash with no backup, key
  destroyed): existing certs keep working until they expire
  (~825 days from issue, by default). But nobody can mint new
  certs, so:

  - You can't add a new peer.
  - You can't renew a cert. When the first leaf cert ages out,
    that node starts failing peer verification on its own
    incoming connections.
  - You can't rotate the CA.

  Recovery from this state is a full re-key: stand up a new CA on
  some other host with `forgather tls init`, mint fresh certs for
  every surviving node, distribute the new `ca.crt` to all peers,
  restart every server. None of the existing data is lost — it's
  a TLS-state reset, not a cluster wipe.

**Mitigation: back up `ca/ca.key` + `ca/ca.crt`.** The CA private
key is the thing that's irreplaceable. Copy `~/.config/forgather/tls/ca/`
to encrypted offline storage (or to a second machine that's not
publicly reachable) so you can mint a replacement leaf cert if A's
disk dies. The CA key is small (≈ 1.7 kB) — easy to keep in a
password manager's secure-notes section, on a USB stick in a safe,
or in a sealed `age`/`gpg` blob in your dotfiles repo.

> **The CA key is a high-trust artifact.** Treat the backup the
> same way you'd treat an SSH agent key — anyone with it can mint
> certs that impersonate any node in your cluster. The backup
> destination must not be reachable from the cluster's threat
> model.

You can also designate a "warm spare" CA holder by mirroring
`~/.config/forgather/tls/ca/` (key included) to a second host via
encrypted rsync or filesystem-level replication. That host can take
over minting immediately if A is permanently lost; the rest of the
cluster doesn't need to be touched because the CA cert (and
therefore every peer's trust bundle) hasn't changed.

> **Don't run `forgather tls init` on more than one node.** Init
> creates a *new* CA. Two CAs in one cluster means peers can't
> validate each other's certs, peer-pull fails closed in both
> directions, and you'll see "fetch failed" against every other
> node in the Cluster view's Nodes tab (and red dots in the
> sidebar Nodes group). Always use `mint` on the CA holder and
> `install` on the peer.

The general N-node setup is three steps:

1. **Choose one node as the CA holder** (typically your developer
   workstation or the first head node). Run `tls init` there
   exactly once — it produces the CA *and* the CA holder's own
   server cert in one step.

2. **For each other node in the cluster**, on the CA holder:
   `forgather tls mint --hostname … --ip … -o /tmp/<node>-tls`
   produces a server cert + key + a copy of `ca.crt`. Distribute
   that directory to the peer (scp), then run
   `forgather tls install --cert … --key … --ca …` on the peer.

3. **Start `forgather server -H 0.0.0.0 --cluster <name>` on every
   node.** mDNS discovery handles the rest; peers find each other
   and immediately speak HTTPS over the shared CA bundle.

The example below walks through a 3-node setup (A is the CA holder;
B and C are peers). Extending to N peers is the same procedure for
each additional node.

### Step 1 — choose a CA holder

Pick exactly one node. Implications you should be comfortable with:

* That node holds `ca/ca.key`, the only key that can mint new certs.
  Anyone with shell access to that file (or to a host that
  bind-mounts it) can produce a cert claiming to be any other node
  in your cluster.
* Every future "add a node" or "renew a cert" operation runs on that
  node. If it's offline, you can't onboard or renew; if it's
  permanently lost without a backup, you're rebuilding the cluster's
  TLS state from scratch (see "What if host A goes away?" above —
  mitigate with a CA-key backup or a warm-spare CA host).
* The CA holder can still be a fully-participating training node;
  the role costs almost nothing day-to-day. It just needs to be the
  one node you don't forget about.

### Step 2 — provision the CA + the CA holder's own cert (host A only)

```bash
forgather tls init
```

`init` auto-discovers this host's hostnames + IPs and bakes them
into the cert's SAN. You don't need to know peers' addresses here
— under the chain-only default, peers will dial A and validate by
CA chain regardless of what SAN is on A's cert.

If browsers / external clients will hit `https://a.lan:8765/` and
those clients enforce strict hostname matching, pass `--hostname`
explicitly: `forgather tls init --hostname a.lan`.

### Step 3 (automated path) — `forgather tls deploy`

If this host has ssh access to every peer, the whole loop is one
command. Two forms, depending on whether the peers are already
running forgather:

**Bootstrap a fresh cluster** (peers have no TLS yet, so they
can't join the cluster yet — chicken-and-egg). Pass the peer
hostnames directly:

```bash
forgather tls init                                  # once, on the CA holder
forgather tls deploy node-b node-c node-d           # ssh to each, mint, install
# Now start servers everywhere:
forgather server -H 0.0.0.0 --cluster mycluster &   # on the CA holder
ssh node-b 'forgather server -H 0.0.0.0 --cluster mycluster &'
# ...same for each peer
```

**Re-deploy to an existing cluster** (peers are already running
and known to the local server's membership table):

```bash
forgather server -H 0.0.0.0 --cluster mycluster &   # if not already up
forgather tls deploy                                # walks the membership table
```

Either form mints a fresh placeholder-SAN cert for each peer, scps
it over, and runs `forgather tls install` on the peer via ssh.
Passwordless ssh is the smooth path; without it ssh prompts for
passwords as usual. `--batch` makes it strict (refuses to prompt).

**Idempotency** is CA-aware: `deploy` reads the peer's existing
`ca.crt` (if any) and compares its SHA-256 to the local CA. Same
fingerprint → silent skip ("already deployed by this CA"). Different
fingerprint → fail with a clear message and require `--force`. No
existing state → proceed normally. Re-running `deploy` on a
correctly-bootstrapped cluster is therefore a no-op.

**Peer runs forgather in Docker?** Pass `--container <name>`. Every
remote command then runs as `docker exec <name> forgather …`, and
the cert files stream into the container via a tar pipe through
`docker exec -i` — no `docker cp` needed, and the host doesn't have
to care where the container's state volume lives.

```bash
forgather tls deploy muthur --container forgather-server
```

For mixed clusters where peers use different container names:
`forgather tls deploy --container-host peer-a=forgather-server --container-host peer-b=fg-prod`.

Other useful flags:

* `--dry-run` — print the plan without minting, scping, or installing.
* `--force` — overwrite an existing-but-different peer CA.
* `--ssh-user USER` — defaults to `$USER`.
* `--ssh-host PEER=HOST` — override the ssh target for a peer (useful
  when the peer's cluster address isn't directly reachable, e.g.
  through a bastion).
* `--container NAME` / `--container-host PEER=NAME` — see above.

The manual flow below is still supported and is the right answer
when ssh isn't available, or when you want to inspect each cert
before installing it.

### Step 3 (manual path) — for each other node, mint + distribute + install

Repeat this block for **every** non-CA-holder node. The 3-node
example covers B and C; add more lines for D, E, …

**On host A** (the CA holder), mint one cert per peer:

```bash
forgather tls mint -o /tmp/b-tls
forgather tls mint -o /tmp/c-tls
# ...one mint per additional peer
```

Each call produces a chain-only-trust cert with a placeholder SAN
(`forgather-peer`, `localhost`, `127.0.0.1`, `::1`). Peers will
validate it by CA chain; the SAN is informational. **You don't need
to know any peer's IP or hostname** — particularly useful when
peers are on a DHCP-issued network you don't control.

If a peer will be reached by a *browser* via its LAN IP/hostname
and you want the browser to validate the SAN, add explicit entries
on that peer's mint:
`forgather tls mint --hostname b.lan --ip 10.0.0.6 -o /tmp/b-tls`.

Each output directory contains `server.crt`, `server.key` (0600),
and a copy of `ca.crt`. The CA private key never leaves A.

**Distribute** each directory to the corresponding peer over a
channel that preserves the 0600 mode on `server.key`:

```bash
scp /tmp/b-tls/server.crt /tmp/b-tls/server.key /tmp/b-tls/ca.crt \
    b.lan:/tmp/b-tls/
scp /tmp/c-tls/server.crt /tmp/c-tls/server.key /tmp/c-tls/ca.crt \
    c.lan:/tmp/c-tls/
# Confirm mode after transfer — expect -rw------- on each server.key.
ssh b.lan 'ls -l /tmp/b-tls/server.key'
ssh c.lan 'ls -l /tmp/c-tls/server.key'
```

Email, Slack DMs, public S3 buckets — anywhere `server.key` could
be read by an unauthorized party — are off-limits. `ca.crt` is safe
to distribute over any channel (it carries no secret), but see the
warning in "Trusting the CA" below: anyone who *trusts* it can be
deceived by certs signed by it.

**On each peer**, install the cert that was minted for it:

```bash
# On host B:
forgather tls install --cert /tmp/b-tls/server.crt \
                      --key  /tmp/b-tls/server.key \
                      --ca   /tmp/b-tls/ca.crt
forgather tls status

# On host C:
forgather tls install --cert /tmp/c-tls/server.crt \
                      --key  /tmp/c-tls/server.key \
                      --ca   /tmp/c-tls/ca.crt
forgather tls status
```

`install` cross-validates that the cert's public key matches the
supplied private key, that the cert chains to the supplied CA, and
that the CA cert is actually a CA. It then writes the key with
mode 0600 from creation (no TOCTOU window), imports the CA into
the trust bundle, populates the SAN list from the cert, and sets
`enabled: true`. The peer can serve TLS but cannot mint new certs
(no CA private key — by design, so a compromised peer can't widen
trust).

### Step 4 — start every server

```bash
# Run on every node (A, B, C, …):
forgather server -H 0.0.0.0 --cluster mycluster
```

mDNS advertisements include a `tls=1` TXT record so peers know
which scheme to use. The peer-pull loop dials `https://...` and
uses the shared CA bundle to validate. Open the Cluster view's
Nodes tab in the webui on any node (or the sidebar Nodes group)
and you should see every other node listed and reachable within
one tick (~5s).

### DiLoCo server

The DiLoCo parameter server picks up the same shared TLS state and
the same bearer-token pattern as `dataset_server` / `inference_server`.
Once a node is provisioned (Step 3) you can launch it like any other
server:

```bash
forgather diloco server -o /path/to/model -n 2 --port 8512 --tls
```

`--tls` forces TLS on for the run; the cert/key come from
`~/.config/forgather/tls/`. The control plane (`/register`, `/status`,
`/control/*`, `/heartbeat`, etc.) requires a bearer token by default —
one is auto-generated and persisted to
`~/.config/forgather/diloco_server/<port>.token` (mode 0600). Clients
discover the token automatically when they connect to a loopback URL
(matches the dataset_server model). For cross-host workers, copy the
token onto the worker host or pass `FORGATHER_DILOCO_SERVER_TOKEN=...`
to the trainer process.

When TLS is enabled and the cluster CA bundle is present, the server
also accepts mTLS handshakes: a client that presents a CA-signed cert
at the TLS handshake is treated as cluster-authenticated and the
bearer check is skipped (same `_PEER_ALLOWED_PATHS` pattern as
`forgather server`). This lets webui proxies and CLI tools talk to a
DiLoCo upstream using only the shared TLS material, with no extra
token plumbing.

#### Bulk data plane (throughput vs security)

Pseudo-gradient and weight transfer can be moved to a separate
cleartext listener via `--bulk-cleartext`:

```bash
# Control on 8512 (TLS + bearer); bulk on a server-picked ephemeral
# port (cleartext, no auth).
forgather diloco server -o /path/to/model -n 2 \
    --port 8512 --tls \
    --bulk-cleartext
```

`--bulk-cleartext` is a single toggle: the bulk listener is always
cleartext, unauthenticated, and bound to an ephemeral port the server
picks itself. Its only purpose is to bypass TLS for throughput on a
trusted LAN — a TLS bulk plane would gain nothing over the control
port, and a bearer token sent over a sniffable socket is theater
(anyone on the wire reads the tensors anyway), so neither is offered.
Workers learn the ephemeral port from the `X-Forgather-Bulk-Url`
header on the `/register` response, which travels over the
TLS-protected control plane — so there's nothing to copy onto worker
command lines and no fixed port to manage. The control port keeps
TLS + bearer regardless, so registration / heartbeat / control actions
stay protected even when the bulk channel is wide open.

**Threat model**: with the bulk plane unauthenticated, an attacker on
the LAN can disrupt a training job (inject garbage pseudo-gradients,
slow it down) but cannot take over the host. Every inbound tensor
blob is loaded with `torch.load(..., weights_only=True)` regardless
of port, which closes the pickle-RCE vector.

#### Audit log

Every register / deregister / eviction / outer-optimizer step /
control action lands as a JSON line in `<output_dir>/diloco_audit.log`.
Best-effort — write failures log a warning and don't fail the
operation. Tokens are never written to this file.

#### Migrating an existing no-auth deployment

If you've been running DiLoCo workers against a pre-#90 server (no
TLS, no bearer), three paths to bring auth online without a flag day:

1. **Loopback-only setup.** Restart the server with no extra flags:
   the per-port token is auto-generated and persisted to
   `~/.config/forgather/diloco_server/<port>.token`. Workers that
   talk to the server via `localhost` / `127.0.0.1` pick the token
   up automatically — no worker changes needed.
2. **Cross-host workers.** Copy the per-port token file to each
   worker host at the same path, or set
   `FORGATHER_DILOCO_SERVER_TOKEN=<token>` in the worker process's
   environment. The webui spawner sets this for managed jobs;
   bare-metal workers need to wire it through their launch script
   (`forgather submit --diloco-server` — and the deprecated
   `diloco worker` alias — propagates it via trainer env vars).
3. **Stay on no-auth.** Add `--no-auth` to the server command. The
   audit log + `weights_only=True` torch.load hardening still apply,
   so an attacker can disrupt training but cannot escalate to RCE.
   Only do this on a network you fully trust.

### Adding a node later

The same Step-3 procedure for one more node, no cluster restart
needed:

```bash
# On host A (the CA holder):
forgather tls mint -o /tmp/d-tls
scp /tmp/d-tls/server.crt /tmp/d-tls/server.key /tmp/d-tls/ca.crt \
    d.lan:/tmp/d-tls/

# On host D:
forgather tls install --cert /tmp/d-tls/server.crt \
                      --key  /tmp/d-tls/server.key \
                      --ca   /tmp/d-tls/ca.crt
forgather server -H 0.0.0.0 --cluster mycluster
```

No hostname/IP arguments are needed because peer-pull validates by
CA chain — D's IP can be whatever the network gives it.

Existing peers pick up the new node via mDNS — no restart required.

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
forgather job scheduler status   # uses the shared bundle automatically
```

## Trusting the CA from a browser

Forgather is typically running on a Linux server you reach over SSH,
while your browser runs on a separate laptop (macOS, Windows, or
another Linux box). The trust install happens *on the laptop*, not
on the server — the server already trusts its own CA. So this is a
two-step process:

1. **Copy the CA cert from the forgather server to the client
   machine** (the laptop running the browser).
2. **Install it into the client's trust store** with whatever
   procedure that OS / browser uses.

> **`ca.crt` is a high-trust artifact.** A machine that trusts this
> CA will accept *any* cert signed by it for *any* hostname an
> attacker can route traffic for. If a colleague's laptop trusts
> your CA, an attacker who steals your CA private key
> (`ca/ca.key`) can mint a cert claiming to be `bank.example.com`
> and that laptop will accept it without warning. Only trust the
> CA on machines you intend to talk to forgather servers from, and
> treat `ca/ca.key` with the same care as an SSH private key
> (0600, never copied, never on shared storage).

### Step 1: Copy `ca.crt` from the server to the client

From the laptop:

```bash
# Easiest: read the cert over the existing SSH session and write
# it to a local file. No key material crosses the wire.
ssh <forgather-host> 'forgather tls export-ca' > forgather-ca.crt

# Equivalent, with scp:
ssh <forgather-host> 'forgather tls export-ca -o /tmp/forgather-ca.crt'
scp <forgather-host>:/tmp/forgather-ca.crt .
```

For containerized servers (`docker/runtime/run.sh`), the same cert
lives inside the state volume:

```bash
ssh <forgather-host> \
    'docker exec forgather-server cat /home/forgather/.config/forgather/tls/ca/ca.crt' \
    > forgather-ca.crt
```

### Step 2: Install into the client's trust store

Pick the section matching the **laptop's** OS (not the server's).

#### macOS (system + Safari + Chrome + Edge)

```bash
# Adds the CA to the System keychain and marks it trusted for SSL.
# Prompts for your sudo password.
sudo security add-trusted-cert -d -r trustRoot \
    -k /Library/Keychains/System.keychain forgather-ca.crt
```

Firefox uses its own store — see the Firefox section below.

#### Linux (system + Chromium + Edge)

Debian / Ubuntu:

```bash
sudo cp forgather-ca.crt /usr/local/share/ca-certificates/forgather-ca.crt
sudo update-ca-certificates
```

Fedora / RHEL / Rocky:

```bash
sudo cp forgather-ca.crt /etc/pki/ca-trust/source/anchors/forgather-ca.crt
sudo update-ca-trust
```

Arch / openSUSE:

```bash
sudo cp forgather-ca.crt /etc/ca-certificates/trust-source/anchors/
sudo update-ca-trust
```

Firefox uses its own store — see the Firefox section below.

#### Windows (system + Edge + Chrome)

PowerShell, run as administrator:

```powershell
Import-Certificate -FilePath forgather-ca.crt `
    -CertStoreLocation Cert:\LocalMachine\Root
```

Or via the GUI: double-click `forgather-ca.crt` → **Install
Certificate** → **Local Machine** → **Place all certificates in the
following store: Trusted Root Certification Authorities**.

Firefox uses its own store — see the Firefox section below.

#### Firefox (every OS)

Firefox does not consult the system trust store. Import the CA
into Firefox's own store:

1. **Preferences** → **Privacy & Security** (or paste
   `about:preferences#privacy` into the URL bar)
2. Scroll to **Certificates** → **View Certificates…**
3. **Authorities** tab → **Import…** → choose `forgather-ca.crt`
4. In the dialog that appears, tick **Trust this CA to identify
   websites** → **OK**

You can verify the import: `about:certificate?cert=...` will show
the CA you just added, with its expiry and SAN.

### Verifying

After installing, restart the browser and load
`https://<forgather-host>:8765/`. A clean lock icon means the CA
was installed correctly; a "Certificate is not valid" warning
means the cert install didn't take (or you've installed the wrong
file — verify the SHA-256 fingerprint with `openssl x509 -in
forgather-ca.crt -noout -fingerprint -sha256`).

### Removing trust

* macOS: **Keychain Access** → **System** → find `Forgather CA <hostname>`
  → right-click → **Delete**.
* Linux: delete the file you copied into the system trust dir, then
  re-run `update-ca-certificates` / `update-ca-trust`.
* Windows: `certmgr.msc` → **Trusted Root Certification Authorities**
  → find the entry → delete.
* Firefox: same dialog as import, then **Delete or Distrust**.

### What `forgather tls trust-system` does

The CLI helper `forgather tls trust-system` prints the same per-OS
commands listed above, computed for the **server's** OS. It exists
for the case where forgather and your browser are on the same
machine (e.g. local development on a laptop). For the headless-
server-plus-remote-laptop case — which is the usual production
shape — read this section instead and run the commands on the
laptop.

## Behind a reverse proxy

If you front your forgather servers with nginx/Caddy/Traefik that
terminates TLS, run forgather itself with `--no-tls --insecure` and
let the proxy handle the cert. Same pattern in a sidecar-style
Docker setup (TLS-terminating proxy container forwards to the
plaintext forgather container on a private network).

## Docker runtime image

The runtime image (`docker/runtime/`) mounts a state volume at
`/home/forgather/.config/forgather` inside the container — that's
the same directory the TLS module uses, so any TLS state lives in
the volume and persists across `docker rm`.

Four deployment patterns:

**0. Bake the TLS state into the image (recommended for clusters):**

The common case for the runtime image is *build once on one machine,
distribute to N peers*. Per-node `forgather tls install` doesn't fit
that shape — it requires the operator to SSH into every peer with a
mint output. Instead, bake the CA + cert directly into the image so
every container that gets pulled from it already shares trust.

```bash
# On the dev workstation where you've already run `forgather tls init`:
TLS_BAKE_FROM_HOST=1 docker/runtime/build.sh forgather:cluster

# Distribute to peers (private registry):
docker tag forgather:cluster myregistry.lan/forgather:cluster
docker push myregistry.lan/forgather:cluster

# Or via docker save | ssh on a trusted channel:
docker save forgather:cluster | ssh peer.lan 'docker load'

# On each peer:
IMAGE=forgather:cluster docker/runtime/run.sh --recreate
```

Every peer container, on first start, copies the baked seed into its
state volume. Same CA + same server identity across all peers means
peer-pull validates without warnings. The dev workstation can also
talk to any of them (it already trusts the CA — that's where the CA
came from).

> **The image is a secret.** A TLS-baked image carries the CA
> private key and the server private key. Anyone who can pull the
> image can mint forgather certs in your cluster's trust domain
> *and* impersonate any of your nodes. Never publish to a public
> registry. Use a private registry on a controlled network, or
> `docker save | ssh`, or removable media.

Alternative bake sources:

```bash
# Explicit path (e.g. CI workspace with a CA built specifically
# for this build):
TLS_BAKE_FROM_DIR=/path/to/tls-state docker/runtime/build.sh

# Multi-build coordination: mint a CA once, bake the *same* CA into
# multiple images (so independently-built images can still talk to
# each other). The CA private key needs to be reachable from the
# CI worker; the leaf cert can be re-issued per build.
```

**Doesn't work** with the bake flow:

* Building two images on two different machines that have each run
  their own `forgather tls init`. Each image carries a different CA,
  and containers won't validate each other's certs. Pick one CA
  holder (a workstation, a build server, or a CI secret store) and
  bake from that source.
* Public images. The CA private key is in the image layers; anyone
  who can `docker pull` can read it.

Three runtime-only patterns (no bake), in order of complexity:

**1. Single-machine HTTPS bring-up (recommended for first-time users):**

```bash
TLS_INIT=1 docker/runtime/run.sh --recreate
```

`TLS_INIT=1` makes the container run `forgather tls init` on first
start (no-op on subsequent starts — the cert is in the named volume
already). The launcher detects TLS state and prints the
`https://…?token=…` URL.

To trust the container's CA from the host browser:

```bash
docker exec forgather-server cat \
    /home/forgather/.config/forgather/tls/ca/ca.crt > /tmp/forgather-ca.crt
# Then: forgather tls trust-system  → instructions for your OS,
#       or import /tmp/forgather-ca.crt into your browser manually.
```

**2. Share TLS state with the host's `forgather tls init`:**

If you've already run `forgather tls init` on the host (the
recommended workflow when the host is also a development machine),
bind-mount your host config dir into the container so both share
the same CA + cert:

```bash
STATE_VOLUME=$HOME/.config/forgather docker/runtime/run.sh --recreate
```

The container sees the host's TLS state and serves HTTPS off the
same CA. The CLI on the host already trusts that CA, so
`forgather job scheduler status` from outside the container Just Works.

**3. Multi-node cluster with one CA holder (no bake):**

Use this when you can't or won't bake the seed into the image
(e.g. you're pulling a public image and adding TLS yourself). Mint
per-host certs on the head node, distribute to peers, install on
each. Heavier than pattern 0; use pattern 0 instead unless you have
a reason not to.

```bash
# Head node (already has TLS provisioned).
forgather tls mint --hostname peer.lan --ip 10.0.0.99 -o /tmp/peer-tls

# Copy /tmp/peer-tls/ to the peer host with 0600 preserved on server.key.
scp /tmp/peer-tls/{server.crt,server.key,ca.crt} peer.lan:/tmp/peer-tls/

# On the peer host, install into a directory the runtime container
# will bind-mount as its state volume.
mkdir -p ~/forgather-state/tls
forgather tls install --cert /tmp/peer-tls/server.crt \
                      --key  /tmp/peer-tls/server.key \
                      --ca   /tmp/peer-tls/ca.crt
# (run with FORGATHER_TLS_DIR=~/forgather-state/tls if you don't want
# it to land in the host's real config dir)

# Launch the container reusing that state.
CLUSTER=mycluster NETWORK=host \
    STATE_VOLUME=$HOME/forgather-state \
    docker/runtime/run.sh --recreate
```

> **`NO_AUTH=1` is independent of TLS.** Set both for the smoke-test
> mode used by `tests/smoke_runtime_multinode.sh`; set neither for
> a production-shape deployment. `TLS_INIT=1` + `NO_AUTH=1`
> together gives encrypted-transport-but-no-token-required, which
> is the right tradeoff for a fully trusted LAN where peers can't
> easily share tokens but can still benefit from preventing
> eavesdroppers.

## Command reference

The shorthand `forgather tls <subcmd>` always operates on the shared
TLS directory (`~/.config/forgather/tls/` on Linux; platform-correct
on macOS/Windows via `platformdirs`). Override via
`FORGATHER_TLS_DIR=/path/to/dir` — useful for tests or per-tenant
isolation.

### `tls init` — provision a CA + server cert for this host

```text
forgather tls init [--hostname NAME …] [--ip IP …] [--ca-name CN] [--force]
```

Creates `ca/ca.{crt,key}` if absent, mints `server.{crt,key}` covering
auto-detected hostnames (`socket.gethostname()`, `socket.getfqdn()`)
and IPs (from `psutil.net_if_addrs()`), then writes `config.yaml`
with `enabled: true`. Auto-discovered SAN entries are capped at 32 —
add more explicitly with repeatable `--hostname` / `--ip`.

* `--ca-name CN` overrides the CA common name (default: `Forgather CA
  <hostname>`).
* `--force` overwrites an existing CA. *Destructive* — every peer's
  trust bundle breaks until you redistribute the new CA.

**When to use:** first-time TLS bring-up on the CA-holding host.
Never run on a peer (use `install` instead — running `init` on a
peer mints a *different* CA that the master won't trust).

### `tls install` — receive a cert minted elsewhere

```text
forgather tls install --cert PATH --key PATH [--ca PATH]
```

Cross-validates that:
1. The cert's public key matches the private key,
2. The cert chains to the supplied CA (if `--ca` is given),
3. The CA cert is actually marked as a CA (BasicConstraints CA:TRUE).

Refuses to install on any mismatch. The private key is written with
0600 perms at creation (no TOCTOU window). Sets `enabled: true` and
populates the SAN list from the installed cert.

**When to use:** the receiving end of `forgather tls mint` on a peer
host. The `--ca` flag is technically optional but recommended —
without it, the peer has no trust anchor for the master's cert.

### `tls mint` — issue a cert for a peer using this host's CA

```text
forgather tls mint --hostname NAME [--hostname NAME …] [--ip IP …] -o DIR
```

Writes `server.crt`, `server.key` (0600), and `ca.crt` into `DIR`.
The directory is created with 0700; the key is created atomically
with 0600 from the start.

**When to use:** the CA holder provisioning a peer. Distribute the
directory via a channel that preserves 0600 perms on `server.key`
(`scp` does; email and shared S3 buckets do not).

### `tls status` — show CA, server cert, and trust state

```text
forgather tls status [--json]
```

Prints the CA subject + expiry, server cert SAN + expiry, trusted
imports, and *diagnostic warnings* covering:

* Cert provisioned but `enabled: false` — you'll need `--tls` per
  invocation; `forgather tls enable` flips the master switch back.
* Server cert expiring within 30 days — run `tls renew --server`.
* SAN gaps: cert SAN doesn't cover one of the host's current
  hostnames/IPs (typical after a hostname change). Suggests
  `tls renew --server --add-hostname … --add-ip …`.

**When to use:** "is TLS actually doing what I think it's doing?"
First place to look when a connection refuses or a peer is shown
unreachable (red dot in the sidebar Nodes group, or in the Cluster
view's Nodes tab).

### `tls renew` — re-issue cert(s) from the existing CA

```text
forgather tls renew [--ca] [--add-hostname NAME …] [--add-ip IP …]
```

* Default: renews the server cert only. Cheap, reversible.
* `--ca`: re-issues the CA itself. *Prompts for `yes` confirmation*.
  Every peer that trusts the old CA breaks until you redistribute
  the new `ca.crt`. See the "Renewal" section above.
* `--add-hostname` / `--add-ip`: union into the SAN before
  re-issuing. Persisted into `config.yaml`.

**When to use:** scheduled rotation, or after a hostname/IP change.
For SAN-only updates without a fresh signature, you still have to
re-issue — there's no "just patch the SAN" path.

### `tls enable` / `tls disable` — flip the master switch

```text
forgather tls enable
forgather tls disable
```

Sets `enabled: true` / `false` in `config.yaml` without touching
any cert files. `disable` is the right tool for "I need HTTP back
for an hour to troubleshoot"; `clean --yes` is the right tool for
"I'm done with this host."

**When to use:** temporarily revert without re-running `init` later.
Servers must be restarted for the change to take effect.

### `tls export-ca` / `tls import-ca` — trust distribution

```text
forgather tls export-ca [-o PATH]
forgather tls import-ca PATH [--name LABEL]
```

* `export-ca` writes `ca/ca.crt` to PATH (default: stdout). Never
  exports the private key.
* `import-ca` validates the cert is a CA, then stores it under
  `trusted/<label>.crt` and rebuilds `ca-bundle.crt`. The bundle is
  what httpx uses for inter-node verification.

**When to use:** sharing CA trust between hosts that don't have a
common CA holder (e.g. two separately-managed clusters that want to
talk to each other).

### `tls trust-system` — OS-level trust instructions

```text
forgather tls trust-system
```

Prints platform-specific commands for adding the local CA to the
system trust store (Linux: `update-ca-certificates` /
`update-ca-trust`; macOS: `security add-trusted-cert`; Windows:
`Import-Certificate`). Firefox uses its own store; instructions are
included separately.

**When to use:** after `tls init`, to make the host's browser
accept the forgather URLs without warnings.

### `tls clean` — wipe everything

```text
forgather tls clean --yes
```

Removes the entire shared TLS directory. *Irreversible.* `--yes` is
required.

**When to use:** decommissioning a host, or a clean-slate redo when
TLS state is too tangled to recover.

## Configuration file format

`config.yaml` under the TLS directory is the single source of truth:

```yaml
enabled: true
auto_on_non_loopback: true
ca_cert: /home/dinalt/.config/forgather/tls/ca/ca.crt
ca_key: /home/dinalt/.config/forgather/tls/ca/ca.key
ca_serial: /home/dinalt/.config/forgather/tls/ca/ca.srl
server_cert: /home/dinalt/.config/forgather/tls/server.crt
server_key: /home/dinalt/.config/forgather/tls/server.key
ca_bundle: /home/dinalt/.config/forgather/tls/ca-bundle.crt
trusted_dir: /home/dinalt/.config/forgather/tls/trusted
san:
  hostnames: [localhost, myhost.lan]
  ips: [127.0.0.1, 10.0.0.5]
validity_days: 825
ca_validity_days: 3650
```

Edit by hand if you need to point at certs in non-default paths
(corporate PKI mounted at `/etc/ssl/...`, for example), or use the
CLI to keep the file consistent.

## Server flags reference

All three servers (`forgather server`, `dataset-server start`,
`inf server`) accept the same shared TLS flag block:

| Flag | Effect |
|---|---|
| `--tls` | Force TLS on for this invocation, overriding `enabled` in `config.yaml`. Cert/key resolved from shared config. |
| `--no-tls` | Force TLS off, overriding `enabled: true`. Servers go back to plain HTTP. |
| `--insecure` | Allow binding a non-loopback host without TLS (cleartext bearer tokens on the wire). Without this, the server refuses to bind. |
| `--tls-cert PATH` | Override the cert path (for BYOC / corporate PKI). |
| `--tls-key PATH` | Override the key path. |

## CLI clients

CLI clients (`forgather job`, `forgather submit`, `forgather gpu`,
`forgather cluster`, `forgather dataset-server
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
* **mTLS for inter-node calls.** Forgather peers authenticate each
  other with mutual TLS: every cluster node presents its CA-signed
  `server.crt` as a client cert when calling `/api/cluster/*_local`
  on another node, and the receiving server's auth middleware
  accepts cert-presence in lieu of a bearer token. See "[Cluster
  inter-node auth (mTLS)](#cluster-inter-node-auth-mtls)" below.
  Browser / CLI clients still authenticate via bearer; they do not
  need to present a cert.
* **Bearer tokens in URLs.** The webui's startup banner and the
  WebSocket TTY-stream URL carry the token as a query parameter
  (`?token=…`). TLS protects the wire, but the token can still
  appear in browser history and uvicorn access logs. Treat the
  banner URL like a password; copy-paste it once into a real
  bookmark rather than leaving it in shell history.
* **Cluster peer trust.** With TLS, peers authenticate each other via
  mutual TLS — the call only succeeds if the caller holds a cert
  signed by the cluster CA. Source-IP matching remains as a
  one-release deprecated compatibility path (see below). The
  operational mitigation against a stolen `server.key` is unchanged:
  keep cluster traffic on a trusted LAN, and tighten filesystem perms
  on `server.key` (0600) on every peer.

## Cluster inter-node auth (mTLS)

Inter-node calls authenticate by mutual TLS, not by source IP. Every
peer-to-peer request presents the calling node's `server.crt` /
`server.key` as a TLS client certificate; the receiving server is
configured with `ssl_cert_reqs=CERT_OPTIONAL` and the cluster CA
bundle, so the handshake validates the cert before the request
reaches application code. The auth middleware then treats cert
presence as proof of cluster membership — a cert signed by the
cluster CA is, by definition, a legitimate peer.

Browser and CLI clients that talk to the server with a bearer token
are unaffected: the TLS listener requests a client cert but does
not require one, so handshakes without a cert succeed and fall
through to the bearer gate.

The path allow-list (`auth._PEER_ALLOWED_PATHS` /
`_PEER_ALLOWED_MUTATIONS`) is the inter-node surface — cert
presence authenticates the caller as a peer, but only those paths
are reachable that way. Anything else still requires a bearer
token even from a peer.

**No operator action is required.** Every node that has run
`forgather tls init` (or received a cert from `forgather tls mint`
plus a copy of the CA via `tls install`) is automatically mTLS-ready.

### Disabling the cert request

The handshake `CertificateRequest` only kicks in when
`cfg.effective_bundle()` resolves to a CA bundle file. A TLS
deployment with only `server.crt` + `server.key` on disk (no
`ca-bundle.crt` or `ca.crt`) skips it and falls back to bearer-only
auth. This is rarely the right call inside a cluster — without the
bundle, the outbound peer-pull also can't validate the other side —
but it's available for one-off TLS-without-cluster setups.
