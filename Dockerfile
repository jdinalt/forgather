# syntax=docker/dockerfile:1.6
#
# Forgather development image — single-user, host-baked.
#
# Base: Ubuntu 24.04 (Python 3.12 ships in-distro). PyTorch wheels
# bundle their own CUDA runtime, so we don't pull in an nvidia/cuda
# base image — GPU access at runtime comes from the host driver via
# nvidia-container-toolkit (`docker run --gpus all ...`).
#
# This image is intentionally NOT user-agnostic — it's a dev container
# scoped to the operator who built it. ``docker/build.sh`` bakes the
# host user's name/UID/GID directly into the image via build args, so
# files created inside the container land owned by the same identity
# on the host clone, with no runtime usermod / gosu / privilege-drop
# dance. (For the user-agnostic, build-once-deploy-everywhere story,
# see ``Dockerfile.runtime``.) Bind-mount your host clone at
# $FORGATHER_REPO and the container will install it editable on first
# start.
#
# See `docker/build.sh` and `docker/run.sh` for convenience wrappers.

FROM ubuntu:24.04

# Host user identity, baked in at build time. ``docker/build.sh``
# overrides these with the host operator's actual values
# (``id -u`` / ``id -g`` / ``id -un``); the defaults below are
# placeholders that let ``docker build`` work without the wrapper.
ARG USER_NAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
ARG VENV_DIR=/opt/forgather/venv
# Set to 1 to install Claude Code (the CLI agent from Anthropic) into
# the image at /usr/bin/claude. Off by default — opt in via
# ``docker/build.sh --claude``. Tooling-only convenience for
# developers who use Claude Code; production builds shouldn't need
# it. The npm package is installed globally so all in-container
# users (including the gosu-dropped one) can invoke ``claude``.
ARG INSTALL_CLAUDE=0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    VIRTUAL_ENV=${VENV_DIR} \
    USER_NAME=${USER_NAME} \
    VENV_DIR=${VENV_DIR} \
    PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
# Split into "Forgather runtime requirements" and "developer convenience".
# BuildKit cache mounts on /var/cache/apt and /var/lib/apt make rebuilds
# instant on this step. The default `docker-clean` apt config wipes the
# cache after each install, so we delete it first. Cache mounts are
# external to the image layer, so the final image is the same size
# either way.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        # Forgather runtime / build deps
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        build-essential \
        ca-certificates \
        git \
        git-lfs \
        graphviz \
        nodejs \
        npm \
        gosu \
        # Developer convenience (a step up from the stripped-down image)
        bash-completion \
        curl \
        wget \
        less \
        man-db \
        vim \
        nano \
        tmux \
        screen \
        htop \
        tree \
        jq \
        ripgrep \
        unzip \
        zip \
        rsync \
        sudo \
        openssh-client \
        gnupg \
        locales \
        tzdata \
        gh \
    && locale-gen en_US.UTF-8 \
    && gosu nobody true

# ---------------------------------------------------------------------------
# Install uv (fast Python package manager) into /usr/local/bin so it's
# available to root and the unprivileged user without touching the
# user's home directory (which is bind-mounted at runtime).
# ---------------------------------------------------------------------------
ADD --chmod=755 https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN UV_INSTALL_DIR=/usr/local/bin /tmp/uv-install.sh \
    && rm -f /tmp/uv-install.sh \
    && uv --version

# ---------------------------------------------------------------------------
# Optionally install Claude Code (Anthropic's CLI agent) so developers
# who use it don't have to re-install on every image rebuild. Off by
# default; enable via ``docker/build.sh --claude`` (sets
# ``--build-arg INSTALL_CLAUDE=1``). Lands at /usr/bin/claude (npm
# global), world-executable so the gosu-dropped user can invoke it.
# ---------------------------------------------------------------------------
RUN if [ "${INSTALL_CLAUDE}" = "1" ]; then \
        echo "[Dockerfile] installing Claude Code (npm global)" && \
        npm install -g @anthropic-ai/claude-code && \
        chmod -R go+rX /usr/lib/node_modules/@anthropic-ai 2>/dev/null || true; \
    fi

# ---------------------------------------------------------------------------
# Create the in-container user with the host operator's UID/GID/name
# (passed in via build args from ``docker/build.sh``) *before* we
# build the venv, so all venv files (~thousands, dominated by
# PyTorch) are owned by the same identity as the host user from the
# start — no runtime usermod, no recursive chown, no gosu drop, no
# permission gymnastics. Ubuntu 24.04 ships with a uid=1000 'ubuntu'
# user; if our UID collides with it we delete the stock account
# first.
# ---------------------------------------------------------------------------
RUN set -eux; \
    if id -u ubuntu >/dev/null 2>&1 && [ "$(id -u ubuntu)" = "${USER_UID}" ]; then \
        userdel -r ubuntu 2>/dev/null || userdel ubuntu; \
    fi; \
    if ! getent group "${USER_GID}" >/dev/null; then \
        groupadd --gid "${USER_GID}" "${USER_NAME}"; \
    fi; \
    if ! id -u "${USER_NAME}" >/dev/null 2>&1; then \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" \
            --shell /bin/bash --create-home "${USER_NAME}"; \
    fi; \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USER_NAME}; \
    chmod 0440 /etc/sudoers.d/90-${USER_NAME}; \
    install -d -o "${USER_UID}" -g "${USER_GID}" /opt/forgather; \
    chmod 0755 /root
# /root is 0700 in the Ubuntu base image, which blocks the unprivileged
# build user from traversing into the uv cache mount at /root/.cache/uv.
# The chmod above opens up just the parent dir; harmless in a container.

# ---------------------------------------------------------------------------
# Everything below this line that touches the venv runs as the
# unprivileged user — that's the same UID/GID/name as the host
# operator, so created files are owned correctly from the start
# and bind-mounted host paths Just Work. We switch back to root
# below for the system-wide /etc/profile.d and entrypoint setup.
# ---------------------------------------------------------------------------
USER ${USER_NAME}

# Build the Forgather virtualenv at /opt/forgather/venv (outside
# /home, so the bind-mounted host home doesn't shadow it).
# /opt/forgather/ is just the venv's parent — there is no in-image
# copy of the repo.
RUN --mount=type=cache,target=/root/.cache/uv,uid=${USER_UID},gid=${USER_GID},sharing=locked \
    uv venv --python python3.12 --seed ${VENV_DIR}

# Install Forgather + every dependency from pyproject.toml. We bind-
# mount the build context read-only and then copy it into a user-
# writable scratch dir at /tmp/src — setuptools insists on writing
# src/<pkg>.egg-info into the source tree during the build, and the
# bind-mounted context inherits root ownership from BuildKit's
# snapshot (rw=true makes the overlay writable but doesn't change
# permissions). No source layer ends up in the image: /tmp/src is
# scoped to this RUN.
#
# The copy + chown run via sudo because BuildKit presents the bind-
# mounted context as root-owned, and any host directory whose mode
# blocks "other" (e.g. 0700 from a 077 umask) is unreadable by the
# unprivileged build user. Reading as root sidesteps that, then the
# chown hands /tmp/src to the user so uv (running as the user) can
# write egg-info into the source tree.
#
# At runtime the entrypoint switches the install to editable mode
# against $FORGATHER_REPO, so this build-time install only seeds the
# heavy dependency layers (PyTorch, transformers, ...) and the
# package metadata gets rewritten on first container start.
#
# Cache mount lives at the user's ~/.cache/uv (uv's documented cache
# path); BuildKit needs explicit uid/gid on the cache volume so the
# unprivileged user can write to it.
RUN --mount=type=cache,target=/root/.cache/uv,uid=${USER_UID},gid=${USER_GID},sharing=locked \
    --mount=type=bind,target=/build-context \
    sudo cp -a /build-context /tmp/src \
    && sudo chown -R ${USER_UID}:${USER_GID} /tmp/src \
    && uv pip install --python ${VENV_DIR}/bin/python /tmp/src \
    && rm -rf /tmp/src

# Recommended: cut-cross-entropy from source for bf16/fp16 numerical
# stability (see docs/getting-started). The pip release lacks the
# accum_e_fp32 / accum_c_fp32 features Forgather relies on. Replaces
# the cut-cross-entropy 25.1.1 wheel installed via pyproject.toml
# above.
RUN --mount=type=cache,target=/root/.cache/uv,uid=${USER_UID},gid=${USER_GID},sharing=locked \
    uv pip install --python ${VENV_DIR}/bin/python \
        "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"

# TensorBoard <= 2.20.0 imports `pkg_resources` at module load,
# but setuptools 82 (Feb 2026) removed pkg_resources entirely.
# Result: `tensorboard --bind_all` exits with
# `ModuleNotFoundError: No module named 'pkg_resources'`.
#
# Upstream fixed this on master (tensorflow/tensorboard@29f809f4)
# by switching to `importlib.metadata` + `packaging`, but no
# tensorboard release contains the fix yet. Backport the patch
# in-place against the installed package — the patch script is
# idempotent and fails loudly if the pre-patch text has moved
# (i.e. tensorboard was upgraded to a release containing the
# fix, in which case drop both this RUN and the patch script).
RUN --mount=type=bind,source=docker/patches/fix_tensorboard_pkg_resources.py,target=/tmp/fix_tb.py \
    ${VENV_DIR}/bin/python /tmp/fix_tb.py

# Switch back to root for the system-wide /etc/profile.d edits and
# the entrypoint COPY into /usr/local/bin.
USER root

# Activate the venv for every interactive shell, regardless of which
# directory the user lands in. /etc/profile.d runs for login shells;
# /etc/bash.bashrc covers non-login interactive shells.
RUN printf '%s\n' \
        '# Forgather venv (baked into image at build time)' \
        'export VIRTUAL_ENV=/opt/forgather/venv' \
        'export PATH="$VIRTUAL_ENV/bin:$PATH"' \
        > /etc/profile.d/10-forgather-venv.sh \
    && chmod 0644 /etc/profile.d/10-forgather-venv.sh \
    && printf '\n# Forgather venv\n. /etc/profile.d/10-forgather-venv.sh\n' \
        >> /etc/bash.bashrc

# One-shot welcome banner. Printed once per shell session, guarded by
# an env-var marker so it doesn't spam every new tab. The bind-host
# section is only relevant under bridge networking — under host
# networking (the default in run.sh) services on 127.0.0.1 inside the
# container are already on the host's loopback.
RUN printf '%s\n' \
        '#!/bin/sh' \
        'if [ -t 1 ] && [ -z "${FORGATHER_DOCKER_BANNER_SEEN:-}" ]; then' \
        '    export FORGATHER_DOCKER_BANNER_SEEN=1' \
        '    cat <<MOTD' \
        '' \
        'Forgather development container' \
        '  venv:        /opt/forgather/venv  (already on PATH; deps only)' \
        '  source:      $FORGATHER_REPO     (bind-mounted host clone)' \
        '' \
        'New here? Start the web UI:' \
        '    forgather server' \
        'then ctrl-click the printed http://localhost:8765/?token=...' \
        'link to open it in your host browser (the token gates auth).' \
        '' \
        'Networking: docker/run.sh defaults to --network host, so' \
        'services bound to 127.0.0.1 inside the container are' \
        'reachable from the host browser as-is. If you launched with' \
        'NETWORK=bridge, every service must bind 0.0.0.0 instead' \
        '(forgather server -H 0.0.0.0, mkdocs --host 0.0.0.0,' \
        ' tensorboard --bind_all, inference --host 0.0.0.0).' \
        'MOTD' \
        'fi' \
        > /etc/profile.d/20-forgather-tips.sh \
    && chmod 0644 /etc/profile.d/20-forgather-tips.sh

# Entrypoint: if FORGATHER_REPO points at a bind-mounted checkout,
# re-install the package in editable mode against that path so the
# user's host-side edits are picked up live.
#
# The entrypoint script is shared with ``Dockerfile.runtime``. There
# the entrypoint runs as root to usermod/gosu-drop into a runtime
# UID; here the image's user IS already the host operator, so the
# entrypoint's phase-1 (root) block is skipped automatically (it
# guards on ``$(id -u) == 0``) and we go straight to the editable-
# install + exec path. That's why we install the entrypoint as root
# (file ownership) but leave the final USER set to the operator,
# unlike the runtime image which ends on USER root.
USER root
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/forgather-entrypoint
ENTRYPOINT ["/usr/local/bin/forgather-entrypoint"]

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD ["bash", "-l"]
