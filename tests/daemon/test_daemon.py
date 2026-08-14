from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from aiosendspin.models.types import PlayerCommand, Roles

from sendspin.daemon.daemon import DaemonArgs, SendspinDaemon
from sendspin.settings import ClientSettings


class _FakeAudioHandler:
    def __init__(self, *, volume: int, muted: bool) -> None:
        self.volume = volume
        self.muted = muted
        self.calls: list[tuple[int, bool]] = []
        self.delay_changes: list[int] = []

    def set_volume(self, volume: int, *, muted: bool) -> None:
        self.calls.append((volume, muted))
        self.volume = volume
        self.muted = muted

    def notify_delay_change(self, delta_us: int) -> None:
        self.delay_changes.append(delta_us)


def _make_daemon(tmp_path: Path, *, settings_volume: int, settings_muted: bool) -> SendspinDaemon:
    settings = ClientSettings(
        _settings_file=tmp_path / "settings.json",
        player_volume=settings_volume,
        player_muted=settings_muted,
    )
    args = DaemonArgs(
        audio_device=SimpleNamespace(index=0, name="Fake Device"),
        client_id="test-client",
        client_name="Test Client",
        settings=settings,
        use_mpris=False,
    )
    return SendspinDaemon(args)


def test_volume_command_uses_audio_handler_muted_state_for_external_volume(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=True)
    daemon._audio_handler = _FakeAudioHandler(volume=41, muted=False)

    payload = SimpleNamespace(
        player=SimpleNamespace(command=PlayerCommand.VOLUME, volume=67, mute=None)
    )

    daemon._handle_server_command(payload)

    assert daemon._audio_handler.calls == [(67, False)]


def test_mute_command_uses_audio_handler_volume_state_for_external_volume(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path, settings_volume=12, settings_muted=False)
    daemon._audio_handler = _FakeAudioHandler(volume=53, muted=False)

    payload = SimpleNamespace(
        player=SimpleNamespace(command=PlayerCommand.MUTE, volume=None, mute=True)
    )

    daemon._handle_server_command(payload)

    assert daemon._audio_handler.calls == [(53, True)]


def test_set_static_delay_uses_applied_tracker_for_delta(tmp_path: Path) -> None:
    """Sync delta is computed from `_static_delay_ms`, not stale settings.

    Reproduces the CLI-override case: settings stays at 0 while the client was
    initialized to 500 from `--static-delay-ms`. A server-initiated delay change
    to 200 must produce delta = -300ms (200 - 500), not -200ms (200 - 0).
    """
    daemon = _make_daemon(tmp_path, settings_volume=25, settings_muted=False)
    daemon._audio_handler = _FakeAudioHandler(volume=25, muted=False)
    daemon._static_delay_ms = 500.0
    # aiosendspin auto-applies before the callback fires, so the client already
    # reports the new value.
    daemon._client = SimpleNamespace(static_delay_ms=200.0)  # type: ignore[assignment]

    payload = SimpleNamespace(
        player=SimpleNamespace(
            command=PlayerCommand.SET_STATIC_DELAY,
            volume=None,
            mute=None,
            static_delay_ms=200,
        )
    )

    # `settings.update` schedules a debounced save via asyncio; wrap in a loop.
    async def run() -> None:
        daemon._handle_server_command(payload)

    asyncio.run(run())

    assert daemon._audio_handler.delay_changes == [-300_000]
    assert daemon._static_delay_ms == 200.0
    assert daemon._settings.static_delay_ms == 200.0


def _export_daemon(tmp_path: Path, *, export_dir: Path | None, use_mpris: bool) -> SendspinDaemon:
    settings = ClientSettings(_settings_file=tmp_path / "settings.json")
    args = DaemonArgs(
        audio_device=SimpleNamespace(index=0, name="Fake Device"),
        client_id="test-client",
        client_name="Test Client",
        settings=settings,
        use_mpris=use_mpris,
        export_dir=export_dir,
    )
    return SendspinDaemon(args)


def test_export_requests_the_metadata_role(tmp_path: Path) -> None:
    daemon = _export_daemon(tmp_path, export_dir=tmp_path / "out", use_mpris=False)
    daemon._exporter = object()

    roles = daemon._client_roles()

    assert Roles.PLAYER in roles
    assert Roles.METADATA in roles
    assert Roles.CONTROLLER not in roles


def test_player_only_without_export_or_mpris(tmp_path: Path) -> None:
    daemon = _export_daemon(tmp_path, export_dir=None, use_mpris=False)

    assert daemon._client_roles() == [Roles.PLAYER]


def test_metadata_role_is_requested_once_with_export_and_mpris(tmp_path: Path) -> None:
    daemon = _export_daemon(tmp_path, export_dir=tmp_path / "out", use_mpris=True)
    daemon._exporter = object()

    roles = daemon._client_roles()

    assert roles.count(Roles.METADATA) <= 1


class _MetadataFakeClient:
    def __init__(self) -> None:
        self.metadata_listeners: list[object] = []

    def add_server_command_listener(self, callback: object):
        return lambda: None

    def add_group_update_listener(self, callback: object):
        return lambda: None

    def add_metadata_listener(self, callback: object):
        self.metadata_listeners.append(callback)

        def unsubscribe() -> None:
            self.metadata_listeners.remove(callback)

        return unsubscribe


class _StubExporter:
    def __init__(self) -> None:
        self.resets = 0

    def handle_metadata(self, payload: object) -> None:
        return

    def notify_reset(self) -> None:
        self.resets += 1


def test_attach_registers_and_detach_removes_the_metadata_listener(tmp_path: Path) -> None:
    daemon = _export_daemon(tmp_path, export_dir=tmp_path / "out", use_mpris=False)
    exporter = _StubExporter()
    daemon._exporter = exporter
    daemon._audio_handler = SimpleNamespace(
        attach_client=lambda client: None, detach_client=lambda: None
    )
    client = _MetadataFakeClient()

    daemon._attach_client(client)
    assert client.metadata_listeners == [exporter.handle_metadata]

    daemon._detach_client()
    assert client.metadata_listeners == []
    assert exporter.resets == 1


def test_no_metadata_listener_without_export(tmp_path: Path) -> None:
    daemon = _export_daemon(tmp_path, export_dir=None, use_mpris=False)
    daemon._audio_handler = SimpleNamespace(
        attach_client=lambda client: None, detach_client=lambda: None
    )
    client = _MetadataFakeClient()

    daemon._attach_client(client)

    assert client.metadata_listeners == []
