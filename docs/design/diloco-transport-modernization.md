# DiLoCo: modernizing the request/response bulk transport

**Status:** WIP design doc, tracking the implementation PR-to-PR. Tier 1
(safetensors wire frame, #184) and the `BulkBytesTransport` seam (#185) have
landed; Tier 1.5 (gRPC) is landing now — see *Tier 1.5 — as built* below. The
remaining follow-up is gRPC TLS/mTLS/bearer parity.

### Tier 1.5 — as built

The gRPC transport landed with two deliberate divergences from the sketch below:

1. **No handler-agnostic "core" extraction.** Rather than restructure the
   synchronization-critical submit/barrier handlers into result-returning cores
   (high risk), the gRPC servicer reuses the **unmodified** HTTP handlers via an
   in-memory `_CapturingHandler` (`grpc_bulk.py`): it reassembles the request
   chunks into the same `[len][header][blob]` frame, drives the handler, and
   captures the framed response bytes + status. The blocking barrier works
   unchanged (the servicer runs in gRPC's thread pool, like the HTTP per-request
   threads). This was lower-risk and fully general (all ops/modes for free).
2. **TLS follows the control-plane posture; auth is bearer-over-TLS, not mTLS.**
   The gRPC listener builds `ssl_server_credentials` from the *same* cert/key as
   the control plane (plumbed to the server as file paths — gRPC needs PEM, not a
   built `ssl_context`), so a TLS cluster runs gRPC over TLS; a cleartext server
   runs gRPC cleartext (trusted-LAN). The worker authenticates by **bearer over
   the encrypted channel**. The original sketch said "mTLS-or-bearer" to mirror
   the HTTP `ssl.CERT_OPTIONAL` listener — but gRPC TLS has **no `CERT_OPTIONAL`
   equivalent**: a client cert is only verified/exposed under
   `require_client_auth=True`, which would reject every non-cert client at the
   handshake. Since the bulk plane is worker-only and the worker always holds the
   per-port token, bearer-over-TLS is the right gate (TLS still gives encryption
   + server auth). `_grpc_security()` is the seam; the bearer rides as call
   metadata only over a secure channel (a token over cleartext is theater).

The negotiation, framing reuse, and HTTP fallback are as designed: `/info`
advertises `transport` + `grpc_endpoint`; the client builds `GrpcBytesTransport`
only when `transport == "grpc"` (a secure channel when the control URL is
`https`); an older server omitting the keys ⇒ HTTP. The `register` param-download
stays on the HTTP control plane.

> TODO(remove once the transport track lands): this is a WIP design doc that
> tracks the modernization PR-to-PR. Once Tier 1 + Tier 1.5 have landed, delete
> this file and fold the as-is description into `docs/trainers/diloco-architecture.md`
> (the Wire Protocol section).

## Goal

DiLoCo has two transport pillars (issue #154). The **collective fast path**
(`CollectiveBackend` over a device-mesh `diloco` axis) is the high-bandwidth,
low-overhead, single-host/intra-cluster regime — Phase 1 + Phase 2a (pipeline)
have landed. This spike is about the **other pillar**: the *elastic
request/response default* — the HTTP-star path that every worker uses unless it
opts into the collective. That default is what serves the workload DiLoCo was
built for: **infrequent, bulky exchanges over heterogeneous, often cross-host or
cross-datacenter links** where latency is cheap (large H) and bandwidth +
elasticity + not melting a single NIC are what matter.

The default works but is primitive on the wire. The goal: make it **fast and
flexible over slow links** without giving up the star topology or any elasticity
feature (workers join/leave mid-run, the server stays the central authority).

## Current state (audited)

The bulk leg today:

- **Serialization is `torch.save` pickle.** `client.py:309` `_serialize_state_dict`
  does `torch.save(state_dict, BytesIO)`; `server.py:343` is the mirror. dtypes
  ride *implicitly inside the pickle* — there is no explicit typed header.
- **Wire frame is `[4-byte len][JSON header][pickle blob]`.** Assembled at
  `client.py:602` (`struct.pack("!I", len(header)) + header + tensor_data`),
  parsed at `server.py:1936`. The JSON header carries only
  `{worker_id, fragment_id?}`.
- **Two full in-RAM copies per exchange.** The worker holds the pseudo-grad dict
  + its serialized blob; the server holds the deserialized dict +
  `_pending_pseudograds[worker_id]` during the barrier. No streaming, no
  chunking, no mid-transfer resume. A large model means a multi-hundred-MB blob
  materialized whole on both ends.
- **Transport is Python stdlib.** `ThreadingHTTPServer` (`server.py:4081`) +
  `urllib` (`client.py:22`). `torch.save`'s zip container is `ZIP_STORED`
  (uncompressed), so the *only* size reduction on the wire is the bf16 cast — no
  gzip/zstd anywhere.
- **A cleartext "bulk listener"** (`server.py:3991`) optionally splits the heavy
  POST off the control port onto an ephemeral, no-TLS, no-auth port, advertised
  to the client via the `X-Forgather-Bulk-Url` header on `/register`. The bearer
  token is deliberately omitted on bulk requests (`client.py:305`).

What is *already good* and must be preserved:

- **The `OuterSyncBackend` seam** (`sync_backend.py:97`). `HttpStarBackend`
  (`sync_backend.py:185`) owns the upload wire-cast (`upload_dtype`/SR) and
  reports on-wire byte sizes (`SyncResult.sent_bytes/recv_bytes`). Backends are
  selected without the worker knowing the transport. The collective + shared-mem
  backends already live behind it.
- **The `/info` negotiation** (`server.py:2794`). The server advertises
  `expected_client_settings` with `settings_authority: "server"` — `sync_every`,
  `dylu`, `upload_dtype`/`upload_sr`/`download_dtype`/`download_sr`,
  `num_fragments`, `heartbeat_timeout`. The worker adopts them verbatim. **This
  is the handshake a transport negotiation extends**: advertise the transport the
  same way the wire-precision knobs are advertised.
- **safetensors is already in the tree** (`sharded_checkpoint.py:16`,
  `server.py:3648` for on-disk checkpoints) — just never used on the wire.

## Where the abstraction seam must sit

`OuterSyncBackend` is the wrong level for a transport swap — it is *outer-step*
granularity (`join`/`synchronize`/`leave`), and `HttpStarBackend` reproduces the
entire HTTP behavior. A second HTTP-but-faster backend would duplicate all of
that. The transport swap needs a **narrower inner seam**: a `BulkBytesTransport`
that only knows how to *move a typed tensor payload and get one back* —
"round-trip these bytes to this endpoint." `HttpStarBackend` (and the
`DiLoCoClient` under it) keeps the protocol (worker_id, barrier semantics,
`/info`, auth) and delegates only the byte-moving to the transport.

```
OuterSyncBackend (outer-step protocol: join/synchronize/leave)
  └─ HttpStarBackend (worker_id, wire-cast, byte accounting, /info adoption)
       └─ DiLoCoClient (endpoints, auth, bulk-URL routing)
            └─ BulkBytesTransport  ← the new seam
                 ├─ HttpBytesTransport   (urllib; the always-available default)
                 ├─ GrpcBytesTransport    (optional; streaming HTTP/2)
                 └─ (ArrowFlightBytesTransport — candidate)
```

Two things travel through this seam and both want a **typed, streamable frame**:
the payload format (Tier 1) and the transport (Tier 1.5). They are separable and
independently shippable.

## Tier 1 — typed, streaming serialization (no new dependency)

Replace the pickle blob with a **safetensors frame**: an explicit dtype/shape
header + raw tensor bytes, zero-copy on load, no pickle (no arbitrary-code
deserialization risk), and the *same format already used for on-disk
checkpoints* — so wire and disk unify.

- **Frame:** keep the `[4-byte len][JSON control header][payload]` envelope (the
  control header still carries `worker_id`/`fragment_id`), but the payload
  becomes a safetensors buffer instead of a pickle blob. The safetensors header
  makes every tensor's dtype/shape explicit on the wire — which is exactly what
  the fp8/fp4 work (#153) wants when it ships packed `uint8` buffers.
- **Streaming framing:** serialize/deserialize **per-tensor** so neither end has
  to hold the whole model twice. safetensors' layout (header offsets + a flat
  byte region) supports writing/reading tensor-by-tensor into a single buffer;
  combined with chunked transfer (Tier 1.5) this removes the
  monolithic-blob-in-RAM problem even before the transport changes.
- **Negotiation:** add `wire_format: "pickle" | "safetensors"` to
  `expected_client_settings`. Default `pickle` for one release (back-compat with
  an older peer), then flip the server default to `safetensors`. Mixed
  old/new peers stay interoperable because the format is negotiated, not assumed.
- **Dependency:** none new — safetensors is already imported. (It should be
  promoted from an implicit/transitive import to an explicit `pyproject.toml`
  dependency as part of this, since the wire path now depends on it.)

Tier 1 is the **no-regrets prerequisite**: it makes the payload a clean typed
frame so Tier 1.5 is a pure transport swap, not a transport+format rewrite. It
is independently useful (zero-copy load, no pickle, explicit dtype) on its own.
The peak-RAM win does *not* land here — `save`/`load` are still whole-buffer —
it arrives with the chunked transport in Tier 1.5.

## Tier 1.5 — streaming RPC transport (gRPC)

Replace `urllib`/`http.server` on the **bulk leg only** with a streaming RPC.
The control plane (`/info`, register, heartbeat, control) can stay on the
existing HTTP server; only the heavy `submit_pseudograd` / `global_params` legs
move. The transport is negotiated through `/info` (`transport: "http" | "grpc"`
+ an advertised endpoint/port), and **HTTP stays the always-available default and
fallback** — a worker talking to a server that does not advertise the gRPC
upgrade uses HTTP transparently. (gRPC is a hard dependency, always importable;
the negotiation is for peer back-compat, not for whether the dep is installed.)

### gRPC vs Arrow Flight

| | gRPC | Arrow Flight |
|---|---|---|
| Base | HTTP/2, protobuf | gRPC underneath (so: a superset) |
| Streaming / chunking / backpressure | yes | yes |
| Zero-copy large arrays | manual (chunk `bytes`) | **purpose-built** (Arrow buffers) |
| Parallel streams for one payload | manual | **built-in** (DoGet/DoPut, endpoints) |
| mTLS / channel credentials | built-in | built-in (inherits gRPC) |
| Dependency weight | `grpcio` (+ `grpcio-tools` build-time) | `pyarrow` (heavier; pulls Arrow) |
| Fit | a clean, general streaming upgrade over urllib | specialized for moving big typed arrays fast |
| Schema friction | hand-write `.proto` for the byte frame | tensor↔Arrow mapping (or ship opaque buffers) |

**Read:** gRPC is the lower-friction, lower-weight, "straight upgrade over
urllib" — topology and elasticity unchanged, mTLS native, and our payload is
already an opaque typed buffer (Tier 1's safetensors frame) so we do **not** need
Arrow's array semantics to get streaming/chunking/backpressure. Arrow Flight's
advantage (zero-copy parallel array streams) is most compelling if the bottleneck
turns out to be array marshalling throughput on a *fast* link — which is exactly
the regime where the **collective** path already wins. For the *slow-link,
cross-host, flexibility* regime the user is targeting, gRPC's streaming + chunking
+ backpressure is the win, and `pyarrow` is a heavier dep for benefit we would not
exercise. **Recommend gRPC; revisit Arrow Flight only if a measured
array-throughput ceiling on fast links justifies the weight.**

### Dependency posture

The DiLoCo server transport is stdlib-only today, but **gRPC is a hard
dependency** — not an optional extra. `grpcio` is already present in the
environment, and a ~6 MB wheel is ~0.1% of what the install already pulls
(torch ~1.7 GB + CUDA wheels ~4.6 GB). The repo also has a deliberate
*no-`[project.optional-dependencies]`* policy (`pyproject.toml` — even
pytest/black/mkdocs sit in the main `dependencies`, to avoid "installed it but
feature X doesn't work" confusion). So:

- `grpcio`, `grpcio-tools`, and `safetensors` all go into main `dependencies`.
  No import guards, no `forgather[grpc]` extra.
- HTTP stays the **default and universal fallback**. gRPC is a **negotiated**
  upgrade advertised via `/info` — negotiation exists purely for *peer
  back-compat* (an older server that doesn't advertise gRPC), not "is the dep
  installed." A gRPC dial failure surfaces as `ConnectionError` into the
  worker's existing reconnect/re-negotiate loop.
- The `.proto` stubs (`bulk_pb2.py`/`bulk_pb2_grpc.py`) are committed/vendored;
  a `proto/generate.sh` regenerates them (grpcio-tools is a main dep, so regen
  just works). Runtime imports only the vendored modules.
- **gRPC supersedes the cleartext bulk listener:** when `grpc_enabled`, the
  ephemeral cleartext bulk listener is not also started — gRPC (TLS-capable
  without urllib overhead) removes the reason that listener existed, and two
  fast paths into the bulk plane is the security smell the role-split prevents.

### Auth / TLS mapping

gRPC has its own credentials model (channel credentials + call metadata). Map the
existing surface onto it: TLS cert/key → `grpc.ssl_server_credentials`; mTLS
(client cert) → `require_client_auth=True` with the CA; bearer token →
per-call metadata (`authorization: Bearer <token>`). The current cleartext
no-auth bulk listener becomes a gRPC server with the *same* posture options
(cleartext for trusted LANs, mTLS for untrusted), advertised via `/info` exactly
as the bulk URL is advertised today.

## Decisions (resolved)

1. **Transport: gRPC** (not Arrow Flight) — the payload is an opaque typed
   buffer, so Arrow's array semantics buy nothing and `pyarrow` is far heavier.
2. **gRPC is a hard dependency, negotiated for back-compat** (see Dependency
   posture). HTTP stays the default + fallback.
3. **Flip the wire-format default to safetensors one release after Tier 1.**
   Tier 1 ships defaulting to `pickle` (interop with an older peer); a one-line
   follow-up flips the server default to `safetensors`.
4. **Compression is out of scope for now.** Orthogonal and cheap to add at the
   frame level later (zstd on the byte region, negotiated), but bf16/fp8 casts
   already shrink the wire and zstd on quantized data has diminishing returns —
   revisit only if slow-link measurements justify it.

## Recommended build order

1. **Tier 1 — safetensors wire frame + per-tensor streaming + `wire_format`
   negotiation.** No new dep; promote safetensors to an explicit dependency.
   Single, locally-reviewable PR behind the `_serialize/_deserialize_state_dict`
   seam.
2. **Refactor: extract `BulkBytesTransport`** from `DiLoCoClient`'s bulk path
   (`HttpBytesTransport` as the first impl, behavior byte-for-byte unchanged).
   Small, mechanical, sets up the swap.
3. **Tier 1.5 — `GrpcBytesTransport`**, negotiated via `/info`, HTTP fallback;
   `grpc_enabled=False` by default (ships dark). Map auth/TLS. The bigger PR.
4. **(Later) compression** at the frame level if slow-link measurements justify
   it.

## Out of scope

- The **collective** fast path (`CollectiveBackend`) — separate pillar, shipped.
- **Tier 3 decoupled** transports (object-store staging, Ray) — different
  elasticity model; revisit if cross-DC staging becomes the priority.
- **fp8/fp4 reduction precision (#153)** — the cast/scaling layer; rides *on top*
  of whatever transport, as packed `uint8`. Tier 1's typed frame helps it; it is
  not part of this track.
