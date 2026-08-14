from __future__ import annotations

from pathlib import Path

import pytest
from aiosendspin.client import AudioFormat, PCMFormat
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import AudioCodec, undefined_field

from sendspin.export import (
    PARTIAL_DIRNAME,
    TMP_DIRNAME,
    Segmenter,
    TrackTags,
    sanitize_component,
)

RATE = 48000
CHANNELS = 2
DEPTH = 16
FRAME_SIZE = CHANNELS * DEPTH // 8

PCM_FORMAT = PCMFormat(sample_rate=RATE, channels=CHANNELS, bit_depth=DEPTH)
AUDIO_FORMAT = AudioFormat(codec=AudioCodec.PCM, pcm_format=PCM_FORMAT)


class _RecordingWriter:
    """Stands in for TrackWriter, capturing PCM without involving PyAV."""

    instances: list[_RecordingWriter] = []

    def __init__(
        self, path: Path, pcm_format: PCMFormat, tags: TrackTags, export_format: str
    ) -> None:
        self.path = path
        self.tags = tags
        self.export_format = export_format
        self.degraded = False
        self.closed = False
        self.data = bytearray()
        self._pcm_format = pcm_format
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        _RecordingWriter.instances.append(self)

    @property
    def frames_written(self) -> int:
        return len(self.data) // self._pcm_format.frame_size

    @property
    def sample_rate(self) -> int:
        return self._pcm_format.sample_rate

    def write(self, pcm: bytes | bytearray) -> None:
        self.data.extend(pcm)

    def close(self) -> None:
        self.closed = True
        self.path.write_bytes(bytes(self.data))

    def abort(self) -> None:
        self.closed = True
        self.path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _reset_writers() -> None:
    _RecordingWriter.instances = []


def _segmenter(tmp_path: Path, *, holdback_us: int = 0, export_format: str = "flac") -> Segmenter:
    return Segmenter(
        export_dir=tmp_path,
        export_format=export_format,
        writer_factory=_RecordingWriter,
        holdback_us=holdback_us,
    )


def _metadata(
    timestamp_us: int,
    *,
    title: object = undefined_field(),
    artist: object = undefined_field(),
    album: object = undefined_field(),
    album_artist: object = undefined_field(),
    year: object = undefined_field(),
    track: object = undefined_field(),
    progress_ms: int | None = None,
    duration_ms: int = 0,
    speed: int = 1000,
) -> SessionUpdateMetadata:
    progress: object = undefined_field()
    if progress_ms is not None:
        progress = Progress(
            track_progress=progress_ms, track_duration=duration_ms, playback_speed=speed
        )
    return SessionUpdateMetadata(
        timestamp=timestamp_us,
        title=title,  # type: ignore[arg-type]
        artist=artist,  # type: ignore[arg-type]
        album=album,  # type: ignore[arg-type]
        album_artist=album_artist,  # type: ignore[arg-type]
        year=year,  # type: ignore[arg-type]
        track=track,  # type: ignore[arg-type]
        progress=progress,  # type: ignore[arg-type]
    )


def _pcm(frames: int, filler: int = 0) -> bytes:
    return bytes([filler]) * (frames * FRAME_SIZE)


def _push_seconds(
    segmenter: Segmenter, start_us: int, seconds: float, *, filler: int = 0, chunk_ms: int = 500
) -> int:
    """Feed ``seconds`` of audio in ``chunk_ms`` chunks; return the next timestamp."""
    timestamp = start_us
    remaining_ms = round(seconds * 1000)
    while remaining_ms > 0:
        step_ms = min(chunk_ms, remaining_ms)
        frames = step_ms * RATE // 1000
        segmenter.push_pcm(timestamp, _pcm(frames, filler), AUDIO_FORMAT)
        timestamp += step_ms * 1000
        remaining_ms -= step_ms
    return timestamp


def _exported(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if p.is_file())


def _partials(tmp_path: Path) -> list[str]:
    partial_dir = tmp_path / PARTIAL_DIRNAME
    if not partial_dir.is_dir():
        return []
    return sorted(p.name for p in partial_dir.iterdir() if p.is_file())


# --- metadata merging -------------------------------------------------------


def test_undefined_fields_keep_previous_values(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="A", album="Album", progress_ms=0, duration_ms=1000)
    )
    # A progress-only update must not wipe the strings.
    segmenter.push_metadata(_metadata(500_000, progress_ms=500, duration_ms=1000))
    _push_seconds(segmenter, 0, 1.0)
    segmenter.stream_end()

    assert _RecordingWriter.instances[0].tags.identity == ("One", "A", "Album")


def test_explicit_none_clears_a_field(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="A", album="Album", progress_ms=0, duration_ms=1000)
    )
    segmenter.push_metadata(_metadata(1_000_000, title="Two", album=None, progress_ms=0))
    _push_seconds(segmenter, 0, 2.0)
    segmenter.stream_end()

    assert _RecordingWriter.instances[-1].tags.identity == ("Two", "A", None)


# --- boundaries -------------------------------------------------------------


def test_boundary_splits_a_chunk_at_the_exact_frame(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(_metadata(0, title="One", artist="A", progress_ms=0, duration_ms=1000))
    # Boundary lands 500ms into a 1s chunk.
    segmenter.push_metadata(
        _metadata(500_000, title="Two", artist="A", progress_ms=0, duration_ms=1000)
    )
    segmenter.push_pcm(0, _pcm(RATE), AUDIO_FORMAT)
    segmenter.stream_end()

    first, second = _RecordingWriter.instances
    assert first.frames_written == RATE // 2
    assert second.frames_written == RATE // 2


def test_boundary_ahead_of_audio_stays_pending(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(_metadata(0, title="One", artist="A", progress_ms=0, duration_ms=5000))
    segmenter.push_metadata(
        _metadata(5_000_000, title="Two", artist="A", progress_ms=0, duration_ms=5000)
    )
    # Only 1s of audio: the second boundary cannot apply yet.
    _push_seconds(segmenter, 0, 1.0)

    assert len(_RecordingWriter.instances) == 1
    assert _RecordingWriter.instances[0].tags.title == "One"


def test_metadata_arriving_after_its_audio_still_splits(tmp_path: Path) -> None:
    # Audio runs 5s ahead of playout, so a server announcing at playout time
    # describes audio already received. The holdback must absorb that.
    segmenter = _segmenter(tmp_path, holdback_us=10_000_000)
    segmenter.push_metadata(_metadata(0, title="One", artist="A", progress_ms=0, duration_ms=3000))
    _push_seconds(segmenter, 0, 6.0)
    assert _RecordingWriter.instances == []  # nothing committed yet

    segmenter.push_metadata(
        _metadata(3_000_000, title="Two", artist="A", progress_ms=0, duration_ms=3000)
    )
    segmenter.stream_end()

    first, second = _RecordingWriter.instances
    assert first.tags.title == "One"
    assert first.frames_written == 3 * RATE
    assert second.tags.title == "Two"
    assert second.frames_written == 3 * RATE


def test_repeat_one_starts_a_new_file(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(_metadata(0, title="One", artist="A", progress_ms=0, duration_ms=2000))
    _push_seconds(segmenter, 0, 2.0)
    # Same track, progress reset to zero: it started over.
    segmenter.push_metadata(_metadata(2_000_000, progress_ms=0, duration_ms=2000))
    _push_seconds(segmenter, 2_000_000, 2.0)
    segmenter.stream_end()

    assert len(_RecordingWriter.instances) == 2
    assert {w.tags.title for w in _RecordingWriter.instances} == {"One"}


def test_paused_progress_update_is_not_a_boundary(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="A", progress_ms=1000, duration_ms=5000)
    )
    _push_seconds(segmenter, 0, 1.0)
    # Paused after one more second of playback.
    segmenter.push_metadata(_metadata(1_000_000, progress_ms=2000, duration_ms=5000, speed=0))
    # Resumed four seconds later, still at the same position.
    segmenter.push_metadata(_metadata(5_000_000, progress_ms=2000, duration_ms=5000))
    _push_seconds(segmenter, 1_000_000, 1.0)
    segmenter.stream_end()

    assert len(_RecordingWriter.instances) == 1


def test_format_change_closes_the_current_file(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(_metadata(0, title="One", artist="A", progress_ms=0, duration_ms=4000))
    _push_seconds(segmenter, 0, 1.0)

    other = AudioFormat(
        codec=AudioCodec.PCM,
        pcm_format=PCMFormat(sample_rate=44100, channels=2, bit_depth=16),
    )
    segmenter.push_pcm(1_000_000, b"\x00" * (44100 * 4), other)
    segmenter.stream_end()

    assert len(_RecordingWriter.instances) == 2
    # Both halves are the same track, so neither can be verified and the second
    # capture is dropped as a duplicate of the first.
    assert _partials(tmp_path) == ["A - One.flac"]
    assert _exported(tmp_path) == []


# --- complete vs partial ----------------------------------------------------


def test_track_bounded_by_clean_boundaries_is_complete(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(segmenter, 0, 2.0)
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(segmenter, 2_000_000, 2.0)
    segmenter.stream_end()

    assert _exported(tmp_path) == ["Band - One.flac"]
    # The last track ended because the stream stopped, so it stays unverified.
    assert _partials(tmp_path) == ["Band - Two.flac"]


def test_joining_mid_track_produces_a_partial(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    # First metadata reports the track already 30s in.
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=30_000, duration_ms=60_000)
    )
    _push_seconds(segmenter, 0, 2.0)
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    segmenter.stream_end()

    assert _exported(tmp_path) == []
    assert _partials(tmp_path) == ["Band - One.flac"]


def test_skipping_before_the_end_produces_a_partial(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=60_000)
    )
    _push_seconds(segmenter, 0, 2.0)
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    segmenter.stream_end()

    assert _exported(tmp_path) == []
    assert _partials(tmp_path) == ["Band - One.flac"]


def test_live_stream_with_unknown_duration_is_judged_by_boundaries(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Radio", progress_ms=0, duration_ms=0)
    )
    _push_seconds(segmenter, 0, 2.0)
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Radio", progress_ms=0, duration_ms=0)
    )
    segmenter.stream_end()

    assert _exported(tmp_path) == ["Radio - One.flac"]


def test_stream_clear_drops_held_audio_and_marks_partial(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path, holdback_us=10_000_000)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(segmenter, 0, 2.0)
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(segmenter, 2_000_000, 1.0)
    segmenter.discontinuity()
    segmenter.stream_end()

    # Everything was still held back, so nothing reached a file.
    assert _RecordingWriter.instances == []
    assert _exported(tmp_path) == []


def test_dropped_audio_demotes_the_file_to_partial(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(segmenter, 0, 1.0)
    segmenter.mark_degraded()
    _push_seconds(segmenter, 1_000_000, 1.0)
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    segmenter.stream_end()

    assert _exported(tmp_path) == []
    assert _partials(tmp_path) == ["Band - One.flac"]


# --- publishing -------------------------------------------------------------


def test_audio_without_a_title_is_discarded(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    _push_seconds(segmenter, 0, 2.0)
    segmenter.stream_end()

    assert _RecordingWriter.instances == []
    assert _exported(tmp_path) == []
    assert not (tmp_path / TMP_DIRNAME).exists()


def test_replaying_a_track_keeps_the_first_capture(tmp_path: Path) -> None:
    for round_index in range(2):
        segmenter = _segmenter(tmp_path)
        segmenter.push_metadata(
            _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=2000)
        )
        _push_seconds(segmenter, 0, 2.0, filler=round_index + 1)
        segmenter.push_metadata(
            _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
        )
        segmenter.stream_end()

    assert _exported(tmp_path) == ["Band - One.flac"]
    assert (tmp_path / "Band - One.flac").read_bytes()[0] == 1
    assert list((tmp_path / TMP_DIRNAME).iterdir()) == []


def test_verified_capture_suppresses_a_later_partial(tmp_path: Path) -> None:
    complete = _segmenter(tmp_path)
    complete.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(complete, 0, 2.0)
    complete.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    complete.stream_end()

    partial = _segmenter(tmp_path)
    partial.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=60_000)
    )
    _push_seconds(partial, 0, 1.0)
    partial.stream_end()

    assert _exported(tmp_path) == ["Band - One.flac"]
    assert _partials(tmp_path) == []


def test_reset_forgets_metadata(tmp_path: Path) -> None:
    segmenter = _segmenter(tmp_path)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(segmenter, 0, 2.0)
    segmenter.reset()

    # Audio after a reset has no metadata to attach to and must be dropped.
    _push_seconds(segmenter, 2_000_000, 2.0)
    segmenter.stream_end()

    assert [w.tags.title for w in _RecordingWriter.instances] == ["One"]


# --- naming -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("AC/DC", "AC_DC"),
        ("a:b*c?d", "a_b_c_d"),
        ("  spaced   out  ", "spaced out"),
        ("...", "Unknown"),
        ("", "Unknown"),
        ("line\nbreak", "line_break"),
    ],
)
def test_sanitize_component(value: str, expected: str) -> None:
    assert sanitize_component(value) == expected


def test_build_filename_falls_back_through_album_artist() -> None:
    assert TrackTags(title="T", album_artist="AA").build_filename("flac") == "AA - T.flac"
    assert TrackTags(title="T").build_filename("aiff") == "Unknown Artist - T.aiff"


def test_build_filename_truncates_on_a_utf8_byte_budget() -> None:
    name = TrackTags(title="曲" * 300, artist="A").build_filename("flac")
    assert len(name.encode("utf-8")) <= 200 + len(".flac")
    assert name.endswith(".flac")


def test_pending_boundary_prevents_mislabelling_earlier_audio(tmp_path: Path) -> None:
    """Audio ahead of a queued boundary must not inherit the next track's tags."""
    segmenter = _segmenter(tmp_path, holdback_us=10_000_000)
    # Joined mid-track: the only metadata we have describes the track starting later.
    segmenter.push_metadata(
        _metadata(2_000_000, title="Next", artist="Band", progress_ms=0, duration_ms=2000)
    )
    # A clear (e.g. at stream start) drops the segment seeded for the outgoing track.
    segmenter.discontinuity()
    _push_seconds(segmenter, 0, 4.0)
    segmenter.stream_end()

    titles = [writer.tags.title for writer in _RecordingWriter.instances]
    assert titles == ["Next"]
    assert _RecordingWriter.instances[0].frames_written == 2 * RATE


def test_stream_clear_keeps_pending_boundaries(tmp_path: Path) -> None:
    """A buffer flush says nothing about boundaries for audio still to come."""
    segmenter = _segmenter(tmp_path, holdback_us=10_000_000)
    segmenter.push_metadata(
        _metadata(0, title="One", artist="Band", progress_ms=0, duration_ms=2000)
    )
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    segmenter.discontinuity()
    _push_seconds(segmenter, 0, 4.0)
    segmenter.stream_end()

    assert [writer.tags.title for writer in _RecordingWriter.instances] == ["One", "Two"]
    assert _exported(tmp_path) == ["Band - One.flac"]
    assert _partials(tmp_path) == ["Band - Two.flac"]


def test_audio_older_than_the_oldest_boundary_is_not_mislabelled(tmp_path: Path) -> None:
    """Joining mid-track: audio before the first known boundary belongs to nobody."""
    segmenter = _segmenter(tmp_path, holdback_us=10_000_000)
    # Metadata for the track starting at 2s arrives while 0-2s is still unidentified.
    segmenter.push_metadata(
        _metadata(2_000_000, title="Two", artist="Band", progress_ms=0, duration_ms=2000)
    )
    segmenter.discontinuity()
    # A further announcement must not retro-tag the unidentified audio as "Two".
    segmenter.push_metadata(
        _metadata(4_000_000, title="Three", artist="Band", progress_ms=0, duration_ms=2000)
    )
    _push_seconds(segmenter, 0, 6.0)
    segmenter.stream_end()

    assert [writer.tags.title for writer in _RecordingWriter.instances] == ["Two", "Three"]
    assert _exported(tmp_path) == ["Band - Two.flac"]
    assert _partials(tmp_path) == ["Band - Three.flac"]
