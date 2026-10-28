"""Load browser presentation assets and serialize their tuning values."""

import json
from importlib.resources import files

from src.config import live


def build_page(interval_milliseconds: int, language: str | None = None) -> str:
    template = (
        files("src.app").joinpath("templates", "live.html").read_text(encoding="utf-8")
    )
    configuration = {
        "transcriptionIntervalMilliseconds": interval_milliseconds,
        "language": language,
        "maximumSessionLanguages": live.MAXIMUM_SESSION_LANGUAGES,
        "sampleRate": live.AUDIO_SAMPLE_RATE,
        "packetSeconds": live.AUDIO_PACKET_SECONDS,
        "maximumWindowSeconds": live.MAXIMUM_TRANSCRIPTION_SECONDS,
        "maximumBufferSeconds": live.MAXIMUM_BUFFER_SECONDS,
        "trackingIntervalMilliseconds": live.TRACKING_INTERVAL_MILLISECONDS,
        "geometryExpirySeconds": live.GEOMETRY_EXPIRY_SECONDS,
        "maximumFrameWidth": live.MAXIMUM_FRAME_WIDTH,
        "maximumFrameHeight": live.MAXIMUM_FRAME_HEIGHT,
        "requestTimeoutMilliseconds": live.REQUEST_TIMEOUT_MILLISECONDS,
        "retryMilliseconds": live.RETRY_MILLISECONDS,
        "jpegQuality": live.JPEG_QUALITY,
        "latencyTargetSeconds": live.LATENCY_TARGET_SECONDS,
        "modelLoadingTimeoutMilliseconds": live.MODEL_LOADING_TIMEOUT_MILLISECONDS,
        "maximumPreviewWords": live.MAXIMUM_PREVIEW_WORDS,
        "captionDisplayMilliseconds": live.CAPTION_DISPLAY_MILLISECONDS,
        "maximumMetricItems": live.MAXIMUM_METRIC_ITEMS,
    }
    serialized = json.dumps(configuration).replace("<", "\\u003c")
    return template.replace("__CONFIGURATION__", serialized)
