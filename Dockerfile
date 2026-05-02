# syntax=docker/dockerfile:1.6
#
# Forgather development image.
#
# Base: Ubuntu 24.04 (Python 3.12 ships in-distro). PyTorch wheels
# bundle their own CUDA runtime, so we don't pull in an nvidia/cuda
# base image — GPU access at runtime comes from the host driver via
# nvidia-container-toolkit (`docker run --gpus all ...`).
#
# Build args USER_NAME / USER_UID / USER_GID let the image carry an
# account that matches the host user, so a bind-mounted home keeps
# correct ownership. Defaults match Ubuntu's first interactive user
# (1000:1000); override at build time when your host user differs:
#
#   docker build \
#     --build-arg USER_NAME=$(id -un) \
#     --build-arg USER_UID=$(id -u) \
#     --build-arg USER_GID=$(id -g) \
#     -t forgather-dev .
#
# See `docker/build.sh` and `docker/run.sh` for convenience wrappers.

FROM ubuntu:24.04

ARG USER_NAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
ARG VENV_DIR=/opt/forgather/venv

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=${VENV_DIR} \
    PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
# Split into "Forgather runtime requirements" and "developer convenience".
# Cleanup of apt lists at the end keeps the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

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
# Build the Forgather virtualenv at a fixed location outside /home so it
# survives a bind-mounted home. We seed the venv from a copy of the repo
# baked into the image; at runtime the entrypoint re-links the editable
# install to the bind-mounted source tree, so user edits show up live.
# ---------------------------------------------------------------------------
RUN mkdir -p /opt/forgather \
    && uv venv --python python3.12 --seed ${VENV_DIR}

# Copy the project (filtered by .dockerignore) into the image. We do
# this in two stages so dependency installation is cached on
# pyproject.toml alone — touching application code below doesn't
# invalidate the (slow) PyTorch download layer.
COPY pyproject.toml /opt/forgather/repo/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DIR}/bin/python -r /opt/forgather/repo/pyproject.toml

# Now copy the rest of the repo and finish the editable install.
COPY . /opt/forgather/repo
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DIR}/bin/python --no-deps -e /opt/forgather/repo

# Recommended: cut-cross-entropy from source for bf16/fp16 numerical
# stability (see docs/getting-started). The pip release lacks the
# accum_e_fp32 / accum_c_fp32 features Forgather relies on.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_DIR}/bin/python \
        "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"

# Prebuild the Forgather server SPA so the in-image copy of the repo
# can serve the web UI without a manual `./build-webui.sh` step. The
# build is fast once node_modules is populated; the npm cache lives
# in the build layer (no runtime hit).
#
# Note: when the user bind-mounts a host-side checkout via
# FORGATHER_REPO, the server runs against *that* tree, whose
# webui/dist/ is independent of this one. The entrypoint warns when
# that dist is missing.
RUN bash /opt/forgather/repo/build-webui.sh

# ---------------------------------------------------------------------------
# Create the in-container user matching the host UID/GID. Ubuntu 24.04
# already ships with a uid=1000 'ubuntu' user; if the requested UID
# collides with it we delete the stock account first so the build
# arg always wins.
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
    chown -R "${USER_UID}:${USER_GID}" /opt/forgather

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

# One-shot welcome banner with the docker-specific gotchas (server
# bind host, webui dist location). Printed once per shell session,
# guarded by a marker file in /tmp so it doesn't spam every new tab.
RUN printf '%s\n' \
        '#!/bin/sh' \
        'if [ -t 1 ] && [ -z "${FORGATHER_DOCKER_BANNER_SEEN:-}" ]; then' \
        '    export FORGATHER_DOCKER_BANNER_SEEN=1' \
        '    cat <<MOTD' \
        '' \
        'Forgather development container' \
        '  venv:        /opt/forgather/venv  (already on PATH)' \
        '  bundled src: /opt/forgather/repo  (used when FORGATHER_REPO is unset)' \
        '' \
        'To reach the server from the host browser, bind to 0.0.0.0' \
        'inside the container — the default 127.0.0.1 is unreachable' \
        'across the container network namespace:' \
        '  forgather server -H 0.0.0.0' \
        '' \
        'For inference / tensorboard jobs that need host access, pass' \
        'the equivalent --host 0.0.0.0 (or set it in the submit form).' \
        'MOTD' \
        'fi' \
        > /etc/profile.d/20-forgather-tips.sh \
    && chmod 0644 /etc/profile.d/20-forgather-tips.sh

# Entrypoint: if FORGATHER_REPO points at a bind-mounted checkout,
# re-install the package in editable mode against that path so the
# user's host-side edits are picked up live.
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/forgather-entrypoint
ENTRYPOINT ["/usr/local/bin/forgather-entrypoint"]

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD ["bash", "-l"]
