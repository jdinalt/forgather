# Forgather runtime Docker image

Documentation for the runtime image moved into the consolidated
[`docker/README.md`](../README.md) — see the **Runtime image —
specifics** section there for everything that used to live here:
design philosophy (immutable by design), source-tree-from-git +
air-gap builds, volumes, multi-node operation, healthcheck, common
overrides, the `--dev` debug opt-in, and distribution.

For the broader multi-node setup (peer discovery, distributed-job
launching, hang diagnosis), see
[`docs/guides/multi-node-training.md`](../../docs/guides/multi-node-training.md).
