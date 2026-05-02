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
# Build the Forgather virtualenv at /opt/forgather/venv (outside /home,
# so the bind-mounted host home doesn't shadow it). /opt/forgather/ is
# just the venv's parent — there is no in-image copy of the repo.
# ---------------------------------------------------------------------------
RUN uv venv --python python3.12 --seed ${VENV_DIR}

# Install Forgather + every dependency from pyproject.toml. We bind-
# mount the build context (filtered by .dockerignore) just for this
# step so uv can build the package — no source layer is baked into
# the image. At runtime the entrypoint switches the install to editable
# mode against $FORGATHER_REPO, so this build-time install only seeds
# the heavy dependency layers (PyTorch, transformers, ...) and the
# package metadata gets rewritten on first container start.
#
# /root/.cache/uv is uv's documented cache path inside Docker builds —
# RUN executes as root, and uv resolves its cache via ~/.cache/uv.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,target=/build-context,rw \
    uv pip install --python ${VENV_DIR}/bin/python /build-context

# Recommended: cut-cross-entropy from source for bf16/fp16 numerical
# stability (see docs/getting-started). The pip release lacks the
# accum_e_fp32 / accum_c_fp32 features Forgather relies on. Replaces
# the cut-cross-entropy 25.1.1 wheel installed via pyproject.toml
# above.
RUN --mount=type=cache,target=/root/.cache/uv \
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
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/forgather-entrypoint
ENTRYPOINT ["/usr/local/bin/forgather-entrypoint"]

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

CMD ["bash", "-l"]
