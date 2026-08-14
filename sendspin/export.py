"""Export each played track to disk as a tagged FLAC or AIFF file.

Audio is tapped after decoding but before the playback engine mangles it for clock
sync, so exported files contain the PCM exactly as the server sent it. Track
boundaries come from ``server/state`` metadata: its ``timestamp`` shares the server
clock with audio chunk timestamps, so a boundary can be cut sample-accurately.

Chunks are held back for :data:`HOLDBACK_US` before being committed to the encoder.
Audio arrives several seconds ahead of playout, so without a holdback a server that
announces metadata at playout time would describe audio we already wrote.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import re
import sys
import threading
from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Final, TypeVar, cast

import av
import numpy as np
from aiosendspin.models.types import UndefinedField

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from aiosendspin.client import AudioFormat, PCMFormat
    from aiosendspin.models.core import ServerStatePayload
    from aiosendspin.models.metadata import SessionUpdateMetadata

logger = logging.getLogger(__name__)


# Container formats accepted by --export-format.
EXPORT_FORMATS: Final = ("flac", "aiff")

# How far behind the newest received chunk PCM is committed to the encoder.
HOLDBACK_US: Final = 10_000_000

# How far a boundary may miss a chunk edge before the split is called unclean.
BOUNDARY_TOLERANCE_US: Final = 1_500_000

# How far captured audio may differ from the reported track duration.
DURATION_TOLERANCE_MS: Final = 2_000

# Backwards progress jump that marks a track restart rather than jitter.
PROGRESS_RESET_TOLERANCE_MS: Final = 1_500

# Bounded work queue; matches the audio worker so overload is bounded the same way.
QUEUE_MAXSIZE: Final = 512

# Subdirectory holding captures that could not be verified as complete.
PARTIAL_DIRNAME: Final = ".partial"

# Subdirectory holding in-progress files, so only finished files are visible.
TMP_DIRNAME: Final = ".tmp"

# UTF-8 byte budget for a filename stem, well under the usual 255-byte limit.
MAX_NAME_BYTES: Final = 200

# Grace period for the export thread to flush its final file on shutdown.
_JOIN_TIMEOUT_SECONDS: Final = 10.0

# PCM bit depth -> FFmpeg sample format. FLAC accepts only s16 and s32.
_SAMPLE_FMT: Final = {16: "s16", 24: "s32", 32: "s32"}

# PCM bit depth -> AIFF codec. AIFF stores big-endian samples.
_AIFF_CODEC: Final = {16: "pcm_s16be", 24: "pcm_s24be", 32: "pcm_s32be"}

_LAYOUT: Final = {1: "mono", 2: "stereo"}

# Vorbis comment keys (FLAC) and FFmpeg generic keys (AIFF -> ID3v2 frames).
_TAG_KEYS: Final = {
    "flac": ("TITLE", "ARTIST", "ALBUM", "ALBUMARTIST", "DATE", "TRACKNUMBER"),
    "aiff": ("title", "artist", "album", "album_artist", "date", "track"),
}

_UNSAFE_RE: Final = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]')

_T = TypeVar("_T")


def _merged(value: _T | None | UndefinedField, current: _T | None) -> _T | None:
    """Apply one metadata diff field: undefined keeps, explicit ``None`` clears."""
    if isinstance(value, UndefinedField):
        return current
    return value


def sanitize_component(value: str) -> str:
    """Make a metadata string safe to use as a path component."""
    cleaned = _UNSAFE_RE.sub("_", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned or "Unknown"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Truncate on a UTF-8 byte budget so long titles cannot exceed name limits."""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", "ignore").rstrip()


@dataclass(slots=True, frozen=True)
class TrackTags:
    """Metadata snapshot describing one track."""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    album_artist: str | None = None
    year: int | None = None
    track: int | None = None
    duration_ms: int = 0
    """Track duration in milliseconds; 0 means unknown (e.g. a live stream)."""
    track_start_us: int | None = None
    """Server clock time at which this track started, when derivable."""

    @property
    def identity(self) -> tuple[str | None, str | None, str | None]:
        """Return the tuple used to tell tracks apart; there is no track id."""
        return (self.title, self.artist, self.album)

    def build_filename(self, extension: str) -> str:
        """Return the flat ``Artist - Title.ext`` filename for this track."""
        artist = sanitize_component(self.artist or self.album_artist or "Unknown Artist")
        title = sanitize_component(self.title or "Unknown Title")
        stem = _truncate_utf8(f"{artist} - {title}", MAX_NAME_BYTES)
        return f"{stem}.{extension}"


@dataclass(slots=True)
class _MetadataState:
    """Running merge of ``server/state`` metadata diffs."""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    album_artist: str | None = None
    year: int | None = None
    track: int | None = None
    progress_ms: int | None = None
    duration_ms: int = 0
    playback_speed: int = 1000

    def apply(self, metadata: SessionUpdateMetadata) -> None:
        """Merge one metadata diff into the running state."""
        self.title = _merged(metadata.title, self.title)
        self.artist = _merged(metadata.artist, self.artist)
        self.album = _merged(metadata.album, self.album)
        self.album_artist = _merged(metadata.album_artist, self.album_artist)
        self.year = _merged(metadata.year, self.year)
        self.track = _merged(metadata.track, self.track)

        if isinstance(metadata.progress, UndefinedField):
            return
        if metadata.progress is None:
            self.progress_ms = None
            self.duration_ms = 0
            self.playback_speed = 1000
            return
        self.progress_ms = metadata.progress.track_progress
        self.duration_ms = metadata.progress.track_duration
        self.playback_speed = metadata.progress.playback_speed

    def snapshot(self, timestamp_us: int) -> TrackTags:
        """Freeze the current state into tags valid at ``timestamp_us``."""
        track_start_us = None
        if self.progress_ms is not None:
            track_start_us = timestamp_us - self.progress_ms * 1000
        return TrackTags(
            title=self.title,
            artist=self.artist,
            album=self.album,
            album_artist=self.album_artist,
            year=self.year,
            track=self.track,
            duration_ms=self.duration_ms,
            track_start_us=track_start_us,
        )

    def reset(self) -> None:
        """Forget everything, so a new connection cannot inherit stale tags."""
        self.title = None
        self.artist = None
        self.album = None
        self.album_artist = None
        self.year = None
        self.track = None
        self.progress_ms = None
        self.duration_ms = 0
        self.playback_speed = 1000


@dataclass(slots=True)
class _PcmChunk:
    """One decoded PCM chunk with its server-clock playout timestamp."""

    server_timestamp_us: int
    data: bytes | bytearray
    pcm_format: PCMFormat

    @property
    def frames(self) -> int:
        """Return the number of whole PCM frames in this chunk."""
        return len(self.data) // self.pcm_format.frame_size

    @property
    def end_us(self) -> int:
        """Return the server clock time just past this chunk's last frame."""
        return self.server_timestamp_us + self.frames * 1_000_000 // self.pcm_format.sample_rate

    def split(self, frames: int) -> tuple[_PcmChunk, _PcmChunk]:
        """Split into a head of ``frames`` frames and the remaining tail."""
        offset = frames * self.pcm_format.frame_size
        tail_us = self.server_timestamp_us + frames * 1_000_000 // self.pcm_format.sample_rate
        return (
            _PcmChunk(self.server_timestamp_us, self.data[:offset], self.pcm_format),
            _PcmChunk(tail_us, self.data[offset:], self.pcm_format),
        )


class TrackWriter:
    """Encode interleaved PCM into a single tagged FLAC or AIFF file."""

    def __init__(
        self,
        path: Path,
        pcm_format: PCMFormat,
        tags: TrackTags,
        export_format: str,
    ) -> None:
        """Open ``path`` for writing and emit the container header and tags."""
        self.path = path
        self.degraded = False
        self._pcm_format = pcm_format
        self._sample_fmt = _SAMPLE_FMT[pcm_format.bit_depth]
        self._layout = _LAYOUT[pcm_format.channels]
        self._frames_written = 0

        if export_format == "flac" and pcm_format.bit_depth == 32:
            logger.warning(
                "FLAC export of 32-bit audio is written as 24-bit; "
                "use --export-format aiff to keep all 32 bits"
            )

        options = {"write_id3v2": "1"} if export_format == "aiff" else {}
        self._container = av.open(str(path), mode="w", format=export_format, options=options)
        # Container metadata must be set before the first packet is muxed.
        self._container.metadata.update(_build_tags(tags, export_format))

        codec = "flac" if export_format == "flac" else _AIFF_CODEC[pcm_format.bit_depth]
        stream_options = {"compression_level": "5"} if export_format == "flac" else {}
        self._stream = cast(
            "av.AudioStream",
            self._container.add_stream(
                codec,
                rate=pcm_format.sample_rate,
                layout=self._layout,
                format=self._sample_fmt,
                options=stream_options,
            ),
        )
        self._stream.time_base = Fraction(1, pcm_format.sample_rate)

    @property
    def frames_written(self) -> int:
        """Return how many PCM frames have been encoded so far."""
        return self._frames_written

    @property
    def sample_rate(self) -> int:
        """Return the sample rate this file was opened with."""
        return self._pcm_format.sample_rate

    def write(self, pcm: bytes | bytearray) -> None:
        """Encode one buffer of interleaved little-endian PCM."""
        frames = len(pcm) // self._pcm_format.frame_size
        if frames == 0:
            return

        payload = self._prepare(pcm, frames)
        try:
            frame = av.AudioFrame(format=self._sample_fmt, layout=self._layout, samples=frames)
            frame.sample_rate = self._pcm_format.sample_rate
            frame.time_base = Fraction(1, self._pcm_format.sample_rate)
            frame.pts = self._frames_written
            frame.planes[0].update(payload)
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        except av.FFmpegError:
            logger.exception("Failed to encode audio for %s", self.path.name)
            self.degraded = True
            return
        self._frames_written += frames

    def close(self) -> None:
        """Flush the encoder and close the container."""
        try:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        except av.FFmpegError:
            logger.exception("Failed to flush encoder for %s", self.path.name)
            self.degraded = True
        finally:
            self._container.close()

    def abort(self) -> None:
        """Close the container and remove the partially written file."""
        with_error = False
        try:
            self._container.close()
        except av.FFmpegError:
            with_error = True
        self.path.unlink(missing_ok=True)
        if with_error:
            logger.debug("Ignored error while aborting %s", self.path.name)

    def _prepare(self, pcm: bytes | bytearray, frames: int) -> bytes | bytearray:
        """Widen 24-bit samples to left-aligned s32 and fix host endianness."""
        samples = frames * self._pcm_format.channels
        if self._pcm_format.bit_depth == 24:
            raw = np.frombuffer(pcm, dtype=np.uint8, count=samples * 3).reshape(samples, 3)
            widened = np.zeros((samples, 4), dtype=np.uint8)
            # Left-align: the FLAC encoder reads the top 24 bits of each s32 sample.
            widened[:, 1:4] = raw
            payload: bytes | bytearray = widened.tobytes()
        else:
            payload = pcm[: samples * (self._pcm_format.bit_depth // 8)]

        if sys.byteorder != "little":
            # FFmpeg's s16/s32 formats are native-endian; Sendspin PCM is little-endian.
            width = 2 if self._pcm_format.bit_depth == 16 else 4
            payload = np.frombuffer(payload, dtype=np.uint8).reshape(-1, width)[:, ::-1].tobytes()
        return payload


def _build_tags(tags: TrackTags, export_format: str) -> dict[str, str]:
    """Map track tags onto the container's tag keys, omitting empty values."""
    keys = _TAG_KEYS[export_format]
    values = (
        tags.title,
        tags.artist,
        tags.album,
        tags.album_artist,
        str(tags.year) if tags.year else None,
        str(tags.track) if tags.track else None,
    )
    return {key: str(value) for key, value in zip(keys, values, strict=True) if value}


@dataclass(slots=True)
class _Segment:
    """One track being captured. The writer is opened on the first committed byte."""

    tags: TrackTags
    started_at_boundary: bool
    writer: TrackWriter | None = None
    degraded: bool = False
    unnameable: bool = False


type WriterFactory = Callable[["Path", "PCMFormat", TrackTags, str], TrackWriter]


class Segmenter:
    """Cut the PCM stream into tracks and write each one to disk.

    Single-threaded and free of its own I/O beyond the injected writer factory, so
    the whole boundary policy can be tested without PyAV or threads.
    """

    def __init__(
        self,
        *,
        export_dir: Path,
        export_format: str,
        writer_factory: WriterFactory | None = None,
        holdback_us: int = HOLDBACK_US,
    ) -> None:
        """Prepare a segmenter writing ``export_format`` files under ``export_dir``."""
        self._export_dir = export_dir
        self._format = export_format
        self._writer_factory: WriterFactory = writer_factory or TrackWriter
        self._holdback_us = holdback_us

        self._meta = _MetadataState()
        self._buffer: deque[_PcmChunk] = deque()
        self._pending: deque[tuple[int, TrackTags]] = deque()
        self._active: _Segment | None = None
        self._identity: tuple[str | None, str | None, str | None] | None = None
        self._progress_ms: int | None = None
        self._progress_at_us: int | None = None
        self._progress_speed = 1000
        self._audio_format: AudioFormat | None = None
        self._last_committed_us: int | None = None
        self._sequence = 0

    def push_metadata(self, metadata: SessionUpdateMetadata) -> None:
        """Merge a metadata diff and queue a track boundary when one is implied."""
        outgoing = self._meta.snapshot(metadata.timestamp)
        self._meta.apply(metadata)
        tags = self._meta.snapshot(metadata.timestamp)
        progress_ms = self._meta.progress_ms
        logger.debug(
            "Metadata at %d: %r progress=%s/%sms",
            metadata.timestamp,
            tags.identity,
            progress_ms,
            tags.duration_ms,
        )

        if tags.identity == (None, None, None):
            self._remember_progress(progress_ms, metadata.timestamp)
            return

        if self._is_new_track(tags, progress_ms, metadata.timestamp):
            boundary_us = (
                tags.track_start_us if tags.track_start_us is not None else (metadata.timestamp)
            )
            if self._active is None and not self._pending:
                # Audio still held back belongs to the track being replaced, not to
                # the one just announced. Tag it before the new tags take over. With
                # a boundary already pending, that audio predates the outgoing track
                # too, so it is left unnamed instead.
                self._active = _Segment(tags=outgoing, started_at_boundary=False)
            self._pending.append((boundary_us, tags))
            # Servers may reorder rarely; keep boundaries in playout order.
            if len(self._pending) > 1:
                self._pending = deque(sorted(self._pending, key=lambda item: item[0]))
            logger.debug("Track boundary queued at %d: %r", boundary_us, tags.identity)
        elif self._active is not None and self._active.tags.identity == tags.identity:
            # Duration often arrives after the track has already started.
            self._active.tags = replace(self._active.tags, duration_ms=tags.duration_ms)

        self._identity = tags.identity
        self._remember_progress(progress_ms, metadata.timestamp)

    def push_pcm(self, server_timestamp_us: int, data: bytes | bytearray, fmt: AudioFormat) -> None:
        """Buffer one decoded PCM chunk and commit whatever has aged past the holdback."""
        if self._audio_format is not None and self._audio_format != fmt:
            logger.debug("Audio format changed mid-stream; closing current export")
            self._drain(_UNBOUNDED_HORIZON)
            self._finalize(ended_at_boundary=False)
        self._audio_format = fmt

        chunk = _PcmChunk(server_timestamp_us, data, fmt.pcm_format)
        if chunk.frames == 0:
            return
        self._buffer.append(chunk)
        self._drain(chunk.end_us - self._holdback_us)

    def discontinuity(self) -> None:
        """Handle ``stream/clear``: buffered audio was discarded, so drop it too.

        Pending boundaries are kept: they are timestamps for audio still to come,
        and a buffer flush says nothing about them.
        """
        logger.debug("Stream cleared; dropping %d held chunks", len(self._buffer))
        self._buffer.clear()
        self._finalize(ended_at_boundary=False)
        self._progress_ms = None

    def stream_end(self) -> None:
        """Commit everything still held and close the current file."""
        self._drain(_UNBOUNDED_HORIZON)
        # A boundary announced for audio we already have means this track really did
        # end here; the stream simply stopped before the next track's audio arrived.
        ended_at_boundary = (
            bool(self._pending)
            and self._last_committed_us is not None
            and self._pending[0][0] <= self._last_committed_us + BOUNDARY_TOLERANCE_US
        )
        self._finalize(ended_at_boundary=ended_at_boundary)
        self._audio_format = None

    def reset(self) -> None:
        """Flush and forget all metadata, for a disconnect or server switch."""
        self.stream_end()
        self._meta.reset()
        self._identity = None
        self._progress_ms = None
        self._pending.clear()

    def mark_degraded(self) -> None:
        """Record that audio was dropped, so the file is not claimed to be complete."""
        if self._active is not None:
            self._active.degraded = True

    def _remember_progress(self, progress_ms: int | None, timestamp_us: int) -> None:
        """Record where playback was, so the next update can be extrapolated from it."""
        self._progress_ms = progress_ms
        self._progress_at_us = timestamp_us if progress_ms is not None else None
        self._progress_speed = self._meta.playback_speed

    def _is_new_track(self, tags: TrackTags, progress_ms: int | None, timestamp_us: int) -> bool:
        """Return whether this metadata update starts a different track."""
        if tags.identity != self._identity:
            return True
        if progress_ms is None or self._progress_ms is None or self._progress_at_us is None:
            return False
        # Extrapolate where the previous update said playback would be by now. A
        # repeat rewinds to zero; a pause simply stops the extrapolation.
        # playback_speed is a multiplier scaled by 1000, so us * speed / 1e6 gives ms.
        elapsed_ms = (timestamp_us - self._progress_at_us) * self._progress_speed // 1_000_000
        expected_ms = self._progress_ms + elapsed_ms
        return progress_ms < expected_ms - PROGRESS_RESET_TOLERANCE_MS

    def _drain(self, horizon_us: int) -> None:
        """Commit buffered chunks that have aged past ``horizon_us``, splitting on boundaries."""
        buffer = self._buffer
        while buffer and buffer[0].end_us <= horizon_us:
            chunk = buffer[0]
            if self._pending and self._pending[0][0] < chunk.end_us:
                boundary_us, tags = self._pending.popleft()
                self._apply_boundary(chunk, boundary_us, tags)
                continue
            buffer.popleft()
            self._write(chunk)

    def _apply_boundary(self, chunk: _PcmChunk, boundary_us: int, tags: TrackTags) -> None:
        """Rotate to a new segment at ``boundary_us``, splitting ``chunk`` if needed."""
        offset_us = boundary_us - chunk.server_timestamp_us
        offset_frames = round(offset_us * chunk.pcm_format.sample_rate / 1_000_000)
        offset_frames = min(max(offset_frames, 0), chunk.frames)

        if offset_frames == 0:
            clean = abs(offset_us) <= BOUNDARY_TOLERANCE_US
            if not clean:
                logger.debug("Track boundary missed audio by %d us", offset_us)
            self._rotate(tags, clean=clean)
            return

        head, tail = chunk.split(offset_frames)
        self._write(head)
        if tail.frames:
            self._buffer[0] = tail
        else:
            self._buffer.popleft()
        self._rotate(tags, clean=True)

    def _rotate(self, tags: TrackTags, *, clean: bool) -> None:
        """Close the active segment and open a new one for ``tags``."""
        self._finalize(ended_at_boundary=clean)
        self._active = _Segment(tags=tags, started_at_boundary=clean)

    def _write(self, chunk: _PcmChunk) -> None:
        """Append a chunk to the active segment, opening its file on first use."""
        segment = self._active
        if segment is None:
            # No boundary seen yet: joined mid-track, or audio resumed after a clear.
            # A boundary still pending means the current tags describe the *next*
            # track, so this audio belongs to one we cannot name.
            tags = TrackTags() if self._pending else self._meta.snapshot(chunk.server_timestamp_us)
            segment = _Segment(tags=tags, started_at_boundary=False)
            self._active = segment

        if segment.unnameable:
            return
        if segment.writer is None:
            if not segment.tags.title:
                logger.debug("Discarding audio with no track title")
                segment.unnameable = True
                return
            segment.writer = self._open_writer(segment.tags, chunk.pcm_format)
            if segment.writer is None:
                segment.unnameable = True
                return
        segment.writer.write(chunk.data)
        self._last_committed_us = chunk.end_us

    def _open_writer(self, tags: TrackTags, pcm_format: PCMFormat) -> TrackWriter | None:
        """Create a writer against a temporary file, or None if that fails."""
        tmp_dir = self._export_dir / TMP_DIRNAME
        self._sequence += 1
        tmp_path = tmp_dir / f"export-{os.getpid()}-{self._sequence}.{self._format}"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            return self._writer_factory(tmp_path, pcm_format, tags, self._format)
        except (OSError, av.FFmpegError, ValueError):
            logger.exception("Failed to start export for %r", tags.identity)
            return None

    def _finalize(self, *, ended_at_boundary: bool) -> None:
        """Close the active segment and move its file into place."""
        segment = self._active
        self._active = None
        if segment is None or segment.writer is None:
            return

        writer = segment.writer
        writer.close()
        if writer.frames_written == 0:
            writer.path.unlink(missing_ok=True)
            return

        captured_ms = writer.frames_written * 1000 // writer.sample_rate
        duration_ms = segment.tags.duration_ms
        if segment.degraded or writer.degraded:
            reason = "audio was dropped"
        elif duration_ms > 0:
            # Capturing the reported duration proves the track is whole, however
            # its boundaries happened to be derived.
            drift_ms = captured_ms - duration_ms
            reason = (
                f"captured {captured_ms}ms of {duration_ms}ms ({drift_ms:+d}ms)"
                if abs(drift_ms) > DURATION_TOLERANCE_MS
                else ""
            )
        elif not segment.started_at_boundary:
            # Unknown duration (live stream): fall back to how the ends were cut.
            reason = "start was not on a track boundary"
        elif not ended_at_boundary:
            reason = "end was not on a track boundary"
        else:
            reason = ""

        if reason:
            logger.debug("Marking %r partial: %s", segment.tags.identity, reason)
        self._publish(writer, segment.tags, partial=bool(reason))

    def _publish(self, writer: TrackWriter, tags: TrackTags, *, partial: bool) -> None:
        """Move a finished temporary file to its destination, keeping the first capture."""
        filename = tags.build_filename(self._format)
        target_dir = self._export_dir / PARTIAL_DIRNAME if partial else self._export_dir
        destination = target_dir / filename
        # A verified capture in the root supersedes any partial of the same name.
        existing = destination.exists() or (partial and (self._export_dir / filename).exists())
        if existing:
            logger.info("Already exported, skipping: %s", filename)
            writer.path.unlink(missing_ok=True)
            return

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            os.replace(writer.path, destination)  # noqa: PTH105
        except OSError:
            logger.exception("Failed to move export into place: %s", filename)
            writer.path.unlink(missing_ok=True)
            return
        logger.info("Exported %s%s", destination, " (partial)" if partial else "")


_UNBOUNDED_HORIZON: Final = 1 << 62


@dataclass(slots=True)
class _PcmItem:
    """Decoded PCM handed over from the audio worker thread."""

    server_timestamp_us: int
    data: bytes | bytearray
    fmt: AudioFormat


@dataclass(slots=True)
class _MetadataItem:
    """A ``server/state`` metadata diff handed over from the event loop."""

    metadata: SessionUpdateMetadata


@dataclass(slots=True)
class _DiscontinuityItem:
    """Buffered audio was discarded upstream (seek)."""


@dataclass(slots=True)
class _StreamEndItem:
    """The audio stream stopped; flush the current file."""


@dataclass(slots=True)
class _ResetItem:
    """The client detached; flush and forget all metadata."""


@dataclass(slots=True)
class _StopItem:
    """Shutdown sentinel for the export thread."""


type _ExportWorkItem = (
    _PcmItem | _MetadataItem | _DiscontinuityItem | _StreamEndItem | _ResetItem | _StopItem
)


class TrackExporter:
    """Write every played track to disk, off the audio path.

    Implements the ``PcmTap`` protocol from :mod:`sendspin.audio_connector`. All
    public methods only enqueue work, so neither the audio worker nor the event
    loop can be blocked by encoding or disk I/O.
    """

    def __init__(self, *, export_dir: Path, export_format: str) -> None:
        """Configure an exporter; call :meth:`start` to spawn its thread."""
        if export_format not in EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {export_format}")
        self._export_dir = export_dir
        self._export_format = export_format
        self._queue: queue.Queue[_ExportWorkItem] | None = None
        self._thread: threading.Thread | None = None
        self._dropped = threading.Event()

    def start(self) -> None:
        """Clean up stale temporary files and start the export thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._clean_tmp_dir()
        self._queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._thread = threading.Thread(target=self._run, name="sendspin-export", daemon=True)
        self._thread.start()
        logger.info("Exporting tracks to %s as %s", self._export_dir, self._export_format)

    async def stop(self) -> None:
        """Signal the export thread to flush its final file and wait for it."""
        queue_obj = self._queue
        thread = self._thread
        self._queue = None
        self._thread = None
        if queue_obj is None:
            return

        try:
            queue_obj.put_nowait(_StopItem())
        except queue.Full:
            logger.warning("Export queue full; forcing shutdown")
            while True:
                try:
                    queue_obj.get_nowait()
                except queue.Empty:
                    break
            with contextlib.suppress(queue.Full):
                queue_obj.put_nowait(_StopItem())

        if thread is not None and thread.is_alive():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, thread.join, _JOIN_TIMEOUT_SECONDS)

    def is_running(self) -> bool:
        """Return whether the export thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def write_pcm(
        self, server_timestamp_us: int, data: bytes | bytearray, fmt: AudioFormat
    ) -> None:
        """Accept decoded PCM from the audio worker thread."""
        self._enqueue(_PcmItem(server_timestamp_us, data, fmt))

    def notify_discontinuity(self) -> None:
        """Accept a buffer-clear notification from the audio worker thread."""
        self._enqueue(_DiscontinuityItem())

    def notify_stream_end(self) -> None:
        """Accept a stream-end notification from the audio worker thread."""
        self._enqueue(_StreamEndItem())

    def notify_reset(self) -> None:
        """Flush and forget all metadata, for a disconnect or server switch."""
        self._enqueue(_ResetItem())

    def handle_metadata(self, payload: ServerStatePayload) -> None:
        """Accept a ``server/state`` metadata update from the event loop."""
        if payload.metadata is None:
            return
        self._enqueue(_MetadataItem(payload.metadata))

    def _enqueue(self, item: _ExportWorkItem) -> None:
        """Best-effort enqueue; a drop demotes the current file to a partial."""
        queue_obj = self._queue
        if queue_obj is None:
            return
        try:
            queue_obj.put_nowait(item)
        except queue.Full:
            if not self._dropped.is_set():
                logger.warning("Export queue full; dropping %s", type(item).__name__)
            self._dropped.set()

    def _clean_tmp_dir(self) -> None:
        """Remove leftovers from a previous run that was killed mid-track."""
        tmp_dir = self._export_dir / TMP_DIRNAME
        if not tmp_dir.is_dir():
            return
        for stale in tmp_dir.iterdir():
            if stale.is_file():
                stale.unlink(missing_ok=True)

    def _run(self) -> None:
        """Export thread body: drain work items into the segmenter."""
        queue_obj = self._queue
        if queue_obj is None:
            return
        segmenter = Segmenter(export_dir=self._export_dir, export_format=self._export_format)
        try:
            while True:
                item = queue_obj.get()
                if self._dropped.is_set():
                    self._dropped.clear()
                    segmenter.mark_degraded()
                if isinstance(item, _StopItem):
                    break
                self._dispatch(segmenter, item)
        except Exception:
            logger.exception("Export thread failed")
        finally:
            try:
                segmenter.stream_end()
            except Exception:
                logger.exception("Failed to flush final export")

    @staticmethod
    def _dispatch(segmenter: Segmenter, item: _ExportWorkItem) -> None:
        """Apply one work item to the segmenter."""
        if isinstance(item, _PcmItem):
            segmenter.push_pcm(item.server_timestamp_us, item.data, item.fmt)
        elif isinstance(item, _MetadataItem):
            segmenter.push_metadata(item.metadata)
        elif isinstance(item, _DiscontinuityItem):
            segmenter.discontinuity()
        elif isinstance(item, _StreamEndItem):
            segmenter.stream_end()
        elif isinstance(item, _ResetItem):
            segmenter.reset()
