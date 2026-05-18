# Forgather Docker images

The full reference for both the dev image and the runtime
(distributable) image — including the CLI listing for every
`build` / `run` flag and env var, multi-node operation,
troubleshooting, and the design rationale for the PUID/PGID
remap and immutable-by-design contract — lives in the docs site:

**[docs/getting-started/docker.md](../docs/getting-started/docker.md)**

That page is also published as part of the Read the Docs build,
so you can browse it without cloning the repo.

## What's in this directory

| File | Purpose |
| - | - |
| `build` | Build the dev image (`Dockerfile`) |
| `run` | Launch / attach the dev container |
| `entrypoint.sh` | Shared entrypoint (`nvidia-smi` probe, PUID/PGID remap on the runtime image, editable install when `FORGATHER_REPO` is set) |
| `_lib.sh` | Shared shell library used by both run-scripts (`run` and `runtime/run.sh`) |
| `runtime/build.sh` | Build the runtime image (`Dockerfile.runtime`) |
| `runtime/run.sh` | Launch / manage the runtime container |
| `patches/` | Backport patches applied at image-build time |

The Dockerfiles themselves are at the repo root: `Dockerfile`
(dev) and `Dockerfile.runtime` (runtime).
