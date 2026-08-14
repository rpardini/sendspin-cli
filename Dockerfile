# Standalone Sendspin (formerly "Resonate") multi-room audio daemon, built from this repo.
# Companion to the kodi-rockchip-gbm container: plays synchronized audio from a Sendspin server.
# https://github.com/Sendspin/sendspin-cli -- built from the working tree, not from PyPI, so
# local changes ship without a release. Independent from the main Dockerfile.
ARG BASE_IMAGE="debian:trixie"

# uv: Astral's Python package/tool manager, straight from the official image.
FROM ghcr.io/astral-sh/uv:latest AS uv

# ---------------------------------------------------------------------------
# Builder: resolve deps from uv.lock and install the project into /opt/sendspin.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
# gcc/libc headers are for sendspin/_volume.c, the optional C volume kernel. It is optional at
# build time (setup.py falls back to numpy), so we force it with SENDSPIN_REQUIRE_C_EXT=1 to
# turn a silent fallback into a build failure.
RUN apt-get -y update && apt-get -y dist-upgrade && \
    apt-get -y install --no-install-recommends ca-certificates gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# uv fetches its own managed CPython (project needs >= 3.12), so nothing here depends on the
# distro's python; the runtime stage copies that interpreter along with the venv.
ENV UV_PYTHON=3.13 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/sendspin \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SENDSPIN_REQUIRE_C_EXT=1

WORKDIR /src

# Dependencies first, without the project itself: this layer only rebuilds when the lockfile or
# packaging metadata changes, not on every source edit. README.md is pulled in because it is the
# declared readme and setuptools reads it while building metadata.
COPY pyproject.toml setup.py uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev --no-editable

# Then the source, and install sendspin itself into the same venv (non-editable: /src is gone
# from the runtime image).
COPY sendspin ./sendspin
RUN uv sync --locked --no-dev --no-editable

# ---------------------------------------------------------------------------
# Runtime: no uv, no compiler, just the venv and its interpreter.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS sendspin

ENV DEBIAN_FRONTEND=noninteractive
# Runtime audio libs: libportaudio2 is sendspin's audio backend; libpulse0 + the alsa->pulse
# plugin let it output to a PulseAudio daemon (see Dockerfile.pulseaudio). "or whatever" indeed.
# alsa-utils is for `sendspin audio-devices list`, which shells out to `aplay -L` to enumerate
# ALSA PCMs. FLAC/AIFF export needs no ffmpeg packages: the PyAV wheel bundles its own.
RUN apt-get -y update && apt-get -y dist-upgrade && \
    apt-get -y install --no-install-recommends \
      ca-certificates libportaudio2 libpulse0 pulseaudio-utils libasound2-plugins alsa-utils && \
    rm -rf /var/lib/apt/lists/*

# The venv hardcodes the absolute path of the interpreter that created it, so both directories
# must land where the builder had them.
COPY --from=builder /opt/uv/python /opt/uv/python
COPY --from=builder /opt/sendspin /opt/sendspin
ENV PATH="/opt/sendspin/bin:${PATH}" \
    VIRTUAL_ENV=/opt/sendspin

# An ALSA config whose `default` is a null sink, for containers with no audio at all. Not active
# unless SENDSPIN_NULL_AUDIO=1 selects it via ALSA_CONFIG_PATH (see docker-entrypoint.sh), because
# a null default that applies by accident turns a real deployment into silence with no error.
# It has to be a full config -- ALSA_CONFIG_PATH replaces alsa.conf -- hence the include.
RUN mkdir -p /etc/alsa && printf '%s\n' \
      '</usr/share/alsa/alsa.conf>' \
      'pcm.!default {' \
      '    type plug' \
      '    slave.pcm { type null }' \
      '}' \
      'ctl.!default { type null }' \
      > /etc/alsa/sendspin-null.conf

COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN sendspin --version && which sendspin && \
    ALSA_CONFIG_PATH=/etc/alsa/sendspin-null.conf sendspin audio-devices list | grep -q default

# ARG invalidates the cache; surfaced as an OCI label for provenance. It is a label only: the
# in-package version stays 0.0.0 (release CI is what stamps pyproject.toml), and rewriting it
# here would put the lockfile out of date and break `uv sync --locked`.
ARG PACKAGE_VERSION="dev"
LABEL org.opencontainers.image.title="sendspin" \
      org.opencontainers.image.description="Sendspin multi-room audio daemon (built from source), companion to kodi-rockchip-gbm" \
      org.opencontainers.image.version="${PACKAGE_VERSION}" \
      org.opencontainers.image.source="https://github.com/Sendspin/sendspin-cli"

# Headless background daemon; append flags at runtime, e.g. --name kodi --audio-device pulse --interface <ip>.
# Needs host networking for mDNS discovery of the Sendspin server:
#   nerdctl run -it --network host <image> --name kodi
#
# The container has no audio of its own: give it a sink, or it exits with "Default audio device
# not found." because PortAudio enumerates nothing.
#   - PulseAudio (what the kodi-rockchip-gbm setup does): reach the server's socket and pass
#     --audio-device pulse.
#   - Host ALSA hardware: --device /dev/snd, then --audio-device <name from `audio-devices list`>.
#   - No playback at all, just capture to disk: -e SENDSPIN_NULL_AUDIO=1, which routes ALSA to a
#     null sink so there is something to open. Export is unaffected -- it taps the decoded PCM
#     ahead of the player:
#       docker run -v ./export:/export -e SENDSPIN_NULL_AUDIO=1 <image> --export-dir /export
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
