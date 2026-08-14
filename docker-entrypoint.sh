#!/bin/sh
# Entrypoint for the sendspin container image (see Dockerfile).
set -eu

# With no /dev/snd and no PulseAudio server, PortAudio enumerates zero devices and the daemon
# exits with "Default audio device not found." before it can do anything. SENDSPIN_NULL_AUDIO=1
# swaps in an ALSA config whose `default` is a null sink, giving PortAudio one openable device.
# Playback goes nowhere; export does not care, since --export-dir taps the decoded PCM before
# the player. Use it for export-only containers and smoke tests, never for actual listening.
if [ "${SENDSPIN_NULL_AUDIO:-0}" = "1" ]; then
    ALSA_CONFIG_PATH=/etc/alsa/sendspin-null.conf
    export ALSA_CONFIG_PATH
fi

# `docker run <image> --name kodi` means `sendspin daemon --name kodi` -- daemon is what this
# image is for -- while an explicit subcommand still reaches the rest of the CLI, e.g.
# `docker run <image> audio-devices list`.
case "${1:-}" in
    player | serve | daemon | audio-devices | servers | clients) ;;
    *) set -- daemon "$@" ;;
esac

exec sendspin "$@"
