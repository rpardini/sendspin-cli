from __future__ import annotations

import asyncio
from pathlib import Path

import av
import numpy as np
import pytest
from aiosendspin.client import AudioFormat, PCMFormat
from aiosendspin.models.core import ServerStatePayload
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import AudioCodec

from sendspin.export import (
    EXPORT_FORMATS,
    PARTIAL_DIRNAME,
    TMP_DIRNAME,
    TrackExporter,
    TrackTags,
    TrackWriter,
)

TAGS = TrackTags(
    title="15 Step",
    artist="Radiohead",
    album="In Rainbows",
    album_artist="Radiohead",
    year=2007,
    track=1,
)

SAMPLE_RATE = 44100
FRAMES = 5000


def _reference_samples(bit_depth: int, count: int) -> np.ndarray:
    """Return sample values within the range the given bit depth can hold."""
    rng = np.random.default_rng(20260814)
    limit = 2 ** (bit_depth - 2)
    return rng.integers(-limit, limit, size=count, dtype=np.int64)


def _pack(samples: np.ndarray, bit_depth: int) -> bytes:
    if bit_depth == 16:
        return samples.astype("<i2").tobytes()
    if bit_depth == 32:
        return samples.astype("<i4").tobytes()
    raw = np.frombuffer(samples.astype("<i4").tobytes(), dtype=np.uint8).reshape(-1, 4)
    return raw[:, :3].tobytes()  # drop the high byte to get little-endian 24-bit


def _decode(path: Path) -> tuple[np.ndarray, av.AudioStream, dict[str, str]]:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        frames = [frame.to_ndarray() for frame in container.decode(audio=0)]
        metadata = dict(container.metadata)
    return np.concatenate(frames, axis=-1), stream, metadata


@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
@pytest.mark.parametrize("bit_depth", [16, 24, 32])
@pytest.mark.parametrize("channels", [1, 2])
def test_round_trip_is_bit_exact(
    tmp_path: Path, export_format: str, bit_depth: int, channels: int
) -> None:
    pcm_format = PCMFormat(sample_rate=SAMPLE_RATE, channels=channels, bit_depth=bit_depth)
    samples = _reference_samples(bit_depth, FRAMES * channels)
    path = tmp_path / f"track.{export_format}"

    writer = TrackWriter(path, pcm_format, TAGS, export_format)
    # Write in several buffers: the encoder must not need block-aligned input.
    packed = _pack(samples, bit_depth)
    step = len(packed) // 3 // pcm_format.frame_size * pcm_format.frame_size
    for offset in range(0, len(packed), step):
        writer.write(packed[offset : offset + step])
    writer.close()

    assert writer.frames_written == FRAMES
    assert writer.sample_rate == SAMPLE_RATE
    assert not writer.degraded

    decoded, stream, _ = _decode(path)
    # PyAV yields (channels, samples); flatten back to interleaved order.
    actual = decoded.T.reshape(-1).astype(np.int64)

    expected = samples if bit_depth != 24 else samples << 8
    if export_format == "flac" and bit_depth == 32:
        # FFmpeg's FLAC encoder tops out at 24 bits per sample.
        expected = (expected >> 8) << 8

    assert stream.codec_context.sample_rate == SAMPLE_RATE
    assert stream.codec_context.channels == channels
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("export_format", EXPORT_FORMATS)
def test_tags_survive_the_round_trip(tmp_path: Path, export_format: str) -> None:
    pcm_format = PCMFormat(sample_rate=SAMPLE_RATE, channels=2, bit_depth=16)
    path = tmp_path / f"track.{export_format}"

    writer = TrackWriter(path, pcm_format, TAGS, export_format)
    writer.write(_pack(_reference_samples(16, 2000), 16))
    writer.close()

    _, _, metadata = _decode(path)
    normalized = {key.lower(): value for key, value in metadata.items()}
    assert normalized["title"] == "15 Step"
    assert normalized["artist"] == "Radiohead"
    assert normalized["album"] == "In Rainbows"
    assert normalized["album_artist"] == "Radiohead"
    assert normalized["date"] == "2007"
    assert normalized["track"] == "1"


def test_missing_tags_are_omitted(tmp_path: Path) -> None:
    pcm_format = PCMFormat(sample_rate=SAMPLE_RATE, channels=2, bit_depth=16)
    path = tmp_path / "track.flac"

    writer = TrackWriter(path, pcm_format, TrackTags(title="Only Title"), "flac")
    writer.write(_pack(_reference_samples(16, 2000), 16))
    writer.close()

    _, _, metadata = _decode(path)
    normalized = {key.lower() for key in metadata}
    assert "title" in normalized
    assert "artist" not in normalized
    assert "album" not in normalized


def test_ragged_tail_bytes_are_ignored(tmp_path: Path) -> None:
    pcm_format = PCMFormat(sample_rate=SAMPLE_RATE, channels=2, bit_depth=16)
    path = tmp_path / "track.flac"

    writer = TrackWriter(path, pcm_format, TAGS, "flac")
    # One byte short of a whole frame at the end.
    writer.write(_pack(_reference_samples(16, 2000), 16) + b"\x00")
    writer.close()

    assert writer.frames_written == 1000


def test_abort_removes_the_file(tmp_path: Path) -> None:
    pcm_format = PCMFormat(sample_rate=SAMPLE_RATE, channels=2, bit_depth=16)
    path = tmp_path / "track.flac"

    writer = TrackWriter(path, pcm_format, TAGS, "flac")
    writer.write(_pack(_reference_samples(16, 2000), 16))
    writer.abort()

    assert not path.exists()


def test_exporter_thread_writes_a_tagged_file(tmp_path: Path) -> None:
    """End-to-end: real thread, real encoder, real files."""
    pcm_format = PCMFormat(sample_rate=SAMPLE_RATE, channels=2, bit_depth=16)
    audio_format = AudioFormat(codec=AudioCodec.PCM, pcm_format=pcm_format)
    one_second = _pack(_reference_samples(16, SAMPLE_RATE * 2), 16)

    def state(timestamp_us: int, title: str) -> ServerStatePayload:
        return ServerStatePayload(
            metadata=SessionUpdateMetadata(
                timestamp=timestamp_us,
                title=title,
                artist="Band",
                album="Album",
                progress=Progress(track_progress=0, track_duration=2000, playback_speed=1000),
            )
        )

    exporter = TrackExporter(export_dir=tmp_path, export_format="flac")
    exporter.start()
    assert exporter.is_running()

    exporter.handle_metadata(state(0, "First"))
    for index in range(2):
        exporter.write_pcm(index * 1_000_000, one_second, audio_format)
    exporter.handle_metadata(state(2_000_000, "Second"))
    for index in range(2, 4):
        exporter.write_pcm(index * 1_000_000, one_second, audio_format)
    exporter.notify_stream_end()

    asyncio.run(exporter.stop())

    assert (tmp_path / "Band - First.flac").is_file()
    # Ended with the stream, but captured its full duration, so still complete.
    assert (tmp_path / "Band - Second.flac").is_file()
    assert not (tmp_path / PARTIAL_DIRNAME).exists()
    assert list((tmp_path / TMP_DIRNAME).iterdir()) == []

    decoded, stream, metadata = _decode(tmp_path / "Band - First.flac")
    assert stream.codec_context.name == "flac"
    assert decoded.size // 2 == SAMPLE_RATE * 2  # two seconds of stereo audio
    assert {key.lower(): value for key, value in metadata.items()}["title"] == "First"
