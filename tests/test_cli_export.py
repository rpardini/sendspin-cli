from __future__ import annotations

from pathlib import Path

import pytest

from sendspin.cli import CLIError, _resolve_export_dir, parse_args
from sendspin.export import EXPORT_FORMATS


def test_daemon_parser_accepts_export_flags() -> None:
    args = parse_args(["daemon", "--export-dir", "/tmp/out", "--export-format", "aiff"])

    assert args.command == "daemon"
    assert args.export_dir == "/tmp/out"
    assert args.export_format == "aiff"


def test_export_flags_default_to_none() -> None:
    args = parse_args(["daemon"])

    assert args.export_dir is None
    assert args.export_format is None


@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_every_supported_format_is_accepted(export_format: str) -> None:
    # cli.py hardcodes the choices to keep PyAV off the startup path; this keeps
    # the two lists from drifting apart.
    args = parse_args(["daemon", "--export-format", export_format])

    assert args.export_format == export_format


def test_unknown_export_format_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["daemon", "--export-format", "mp3"])


def test_tui_parser_rejects_export_dir() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--export-dir", "/tmp/out"])


def test_resolve_export_dir_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "captures"

    resolved = _resolve_export_dir(str(target))

    assert resolved == target
    assert target.is_dir()


def test_resolve_export_dir_returns_none_when_unset() -> None:
    assert _resolve_export_dir(None) is None
    assert _resolve_export_dir("") is None


def test_resolve_export_dir_expands_a_home_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = _resolve_export_dir("~/captures")

    assert resolved == tmp_path / "captures"


def test_resolve_export_dir_reports_an_unusable_path(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")

    with pytest.raises(CLIError, match="Cannot create export directory"):
        _resolve_export_dir(str(blocker / "captures"))
