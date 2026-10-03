import json
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
MAXIMUM_LIVE_DURATION_SECONDS = 32.0
MAXIMUM_LIVE_VIDEO_PIXELS = 1280 * 720
LIVE_VIDEO_FRAME_RATE = 25
LIVE_PROBE_TIMEOUT_SECONDS = 20
LIVE_NORMALIZATION_TIMEOUT_SECONDS = 60
DURATION_LIMIT_PROBE_MARGIN_SECONDS = 1.0


class MediaError(RuntimeError):
    """Raised when an input cannot be processed as a video with audio."""


@dataclass(frozen=True)
class MediaInfo:
    """Video stream metadata needed by the renderer."""

    duration: float
    width: int
    height: int
    frame_rate: float
    has_video: bool
    has_audio: bool


@dataclass(frozen=True)
class VideoFrame:
    """A decoded frame and its presentation timestamp."""

    timestamp: float
    image: object


def _run(
    command: list[str], timeout_seconds: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one FFmpeg command and convert tool failures to ``MediaError``."""
    try:
        executable = shutil.which(command[0])
        if executable is None:
            raise FileNotFoundError(command[0])
        # Absolute executable + non-inheritable Python descriptors allow posix_spawn;
        # fork can deadlock in OpenBLAS's atfork handler during concurrent masks.
        return subprocess.run(
            command,
            executable=executable,
            close_fds=False,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise MediaError("FFmpeg/FFprobe is required on PATH") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "media command failed"
        raise MediaError(message) from error
    except subprocess.TimeoutExpired as error:
        raise MediaError(f"media command exceeded {timeout_seconds} seconds") from error


def inspect_media(input_path: Path) -> MediaInfo:
    """Validate that an input has usable video and audio streams.

    Args:
        input_path: Source media file to inspect.

    Returns:
        Duration, dimensions, and frame rate from the primary video stream.

    Raises:
        MediaError: If the source is missing, unsupported, or lacks audio/video.
    """
    if not input_path.is_file():
        raise MediaError(f"input does not exist: {input_path}")

    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), None
    )
    if video is None or audio is None:
        raise MediaError("input must contain both video and audio streams")

    numerator, denominator = video.get("r_frame_rate", "0/1").split("/")
    frame_rate = float(numerator) / float(denominator) if float(denominator) else 0.0
    duration = float(
        video.get("duration") or payload.get("format", {}).get("duration") or 0.0
    )
    if frame_rate <= 0 or duration <= 0:
        raise MediaError("input has no usable duration or frame rate")
    info = MediaInfo(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        frame_rate=frame_rate,
        has_video=True,
        has_audio=True,
    )
    LOGGER.info(
        "input is %.2fs, %dx%d at %.3f fps",
        info.duration,
        info.width,
        info.height,
        info.frame_rate,
    )
    return info


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Extract a 16 kHz mono PCM WAV audio stream with FFmpeg.

    Args:
        input_path: Source video containing audio.
        output_path: WAV path to create.

    Returns:
        The created WAV path.

    Raises:
        MediaError: If FFmpeg cannot extract the selected audio stream.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    LOGGER.info("extracted normalized audio to %s", output_path)
    return output_path


def normalize_camera_recording(
    input_path: Path, output_path: Path, start_seconds: float = 0.0
) -> Path:
    """Create a timestamped MP4 from a browser camera recording, optionally trimmed.

    Browser recordings can omit WebM duration metadata even after ``stop``. Re-encoding
    supplies presentation timestamps required by the shared video recognizer.
    """
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise MediaError("recording start must be finite and nonnegative")
    if start_seconds >= MAXIMUM_LIVE_DURATION_SECONDS:
        raise MediaError("camera recording exceeds the duration limit")
    dimensions_result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(input_path),
        ],
        timeout_seconds=LIVE_PROBE_TIMEOUT_SECONDS,
    )
    video = next(iter(json.loads(dimensions_result.stdout).get("streams", [])), {})
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if width <= 0 or height <= 0 or width * height > MAXIMUM_LIVE_VIDEO_PIXELS:
        raise MediaError("camera recording has missing or oversized video dimensions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-r",
        str(LIVE_VIDEO_FRAME_RATE),
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
    ]
    if start_seconds:
        command.extend(["-ss", str(start_seconds)])
    remaining_duration = MAXIMUM_LIVE_DURATION_SECONDS - start_seconds
    command.extend(
        ["-t", str(remaining_duration + DURATION_LIMIT_PROBE_MARGIN_SECONDS)]
    )
    command.append(str(output_path))
    _run(command, timeout_seconds=LIVE_NORMALIZATION_TIMEOUT_SECONDS)
    duration_result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(output_path),
        ],
        timeout_seconds=LIVE_PROBE_TIMEOUT_SECONDS,
    )
    duration = float(
        json.loads(duration_result.stdout).get("format", {}).get("duration", 0)
    )
    if not 0 < duration <= remaining_duration:
        raise MediaError(
            "camera recording has missing duration or exceeds the duration limit"
        )
    LOGGER.info("normalized browser recording to %s", output_path)
    return output_path


def read_video_frames(input_path: Path) -> list[VideoFrame]:
    """Decode video frames while retaining FFprobe presentation timestamps.

    Args:
        input_path: Source video to decode.

    Returns:
        Ordered decoded frames on the original presentation timeline, backed by
        a temporary memory map so long videos do not retain all images in RAM.

    Raises:
        MediaError: If no frames can be decoded.
    """
    presentation_timestamps = video_presentation_timestamps(input_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise MediaError(f"OpenCV cannot open video: {input_path}")

    timestamps: list[float] = []
    frame_shape = None
    frame_rate = capture.get(cv2.CAP_PROP_FPS) or 25.0
    # ponytail: disk-backed full frames; stream face crops if temporary disk usage matters.
    with tempfile.TemporaryFile() as frame_file:
        try:
            while True:
                success, image = capture.read()
                if not success:
                    break
                frame_index = len(timestamps)
                timestamp = (
                    presentation_timestamps[frame_index]
                    if frame_index < len(presentation_timestamps)
                    else capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )
                if timestamp <= 0 and frame_index:
                    timestamp = frame_index / frame_rate
                if frame_shape is not None and image.shape != frame_shape:
                    raise MediaError("video changes frame dimensions during decoding")
                frame_shape = image.shape
                image.tofile(frame_file)
                timestamps.append(timestamp)
        finally:
            capture.release()
        if not timestamps:
            raise MediaError("input contains no decodable video frames")
        frame_file.flush()
        images = np.memmap(
            frame_file, dtype=np.uint8, mode="r", shape=(len(timestamps), *frame_shape)
        )
    return [
        VideoFrame(timestamp, image)
        for timestamp, image in zip(timestamps, images, strict=True)
    ]


def video_presentation_timestamps(input_path: Path) -> list[float]:
    """Return the primary video stream's presentation timestamps.

    Args:
        input_path: Source video to inspect.

    Returns:
        Ordered presentation timestamps for decoded video frames.

    Raises:
        MediaError: If FFprobe cannot inspect the video stream.
    """
    timestamp_result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(input_path),
        ]
    )
    return [
        float(frame["best_effort_timestamp_time"])
        for frame in json.loads(timestamp_result.stdout).get("frames", [])
        if frame.get("best_effort_timestamp_time") is not None
    ]
