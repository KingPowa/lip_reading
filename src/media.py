import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import ANALYSIS_FRAME_INTERVAL


LOGGER = logging.getLogger(__name__)


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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one FFmpeg command and convert tool failures to ``MediaError``."""
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as error:
        raise MediaError("FFmpeg/FFprobe is required on PATH") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "media command failed"
        raise MediaError(message) from error


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
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise MediaError("input must contain both video and audio streams")

    numerator, denominator = (video.get("r_frame_rate", "0/1").split("/"))
    frame_rate = float(numerator) / float(denominator) if float(denominator) else 0.0
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
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
    LOGGER.info("input is %.2fs, %dx%d at %.3f fps", info.duration, info.width, info.height, info.frame_rate)
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


def read_video_frames(input_path: Path) -> list[VideoFrame]:
    """Decode video frames while retaining FFprobe presentation timestamps.

    Args:
        input_path: Source video to decode.

    Returns:
        Ordered decoded frames on the original presentation timeline.

    Raises:
        MediaError: If no frames can be decoded.
    """
    presentation_timestamps = video_presentation_timestamps(input_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise MediaError(f"OpenCV cannot open video: {input_path}")

    frames: list[VideoFrame] = []
    frame_index = 0
    frame_rate = capture.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        success, image = capture.read()
        if not success:
            break
        timestamp = (
            presentation_timestamps[frame_index]
            if frame_index < len(presentation_timestamps)
            else capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        )
        if timestamp <= 0 and frame_index:
            timestamp = frame_index / frame_rate
        if frame_index % ANALYSIS_FRAME_INTERVAL == 0:
            frames.append(VideoFrame(timestamp=timestamp, image=image))
        frame_index += 1
    capture.release()
    if not frames:
        raise MediaError("input contains no decodable video frames")
    return frames


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
