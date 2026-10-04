"""Output video and accessible caption artifacts."""

import json
import logging
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.app.rendering.colors import speaker_color_bgr
from src.app.rendering.payloads import caption_payload
from src.config.rendering import (
    BODY_HIGHLIGHT_OPACITY,
    CAPTION_BACKGROUND,
    CAPTION_FONT_PATH,
    CAPTION_FONT_SIZE,
    CAPTION_HORIZONTAL_MARGIN,
    CAPTION_LINE_HEIGHT,
)
from src.core.contracts import CaptionEvent, FaceTrack
from src.core.media import video_presentation_timestamps
from src.core.vision.faces import body_box, box_at

LOGGER = logging.getLogger(__name__)

VIEWER_STYLE = """
<style>
body { background: #111827; color: #f9fafb; font: 16px system-ui, sans-serif; margin: 0; }
main { display: grid; gap: 24px; grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr); margin: auto; max-width: 1400px; padding: 24px; }
h1 { font-size: 1.25rem; grid-column: 1 / -1; margin: 0; }
#player { position: relative; }
video { display: block; max-width: 100%; width: 100%; }
#faces { inset: 0; pointer-events: none; position: absolute; }
.speaker { background: color-mix(in srgb, var(--speaker-color) 22%, transparent); border: 4px solid var(--speaker-color); box-sizing: border-box; cursor: help; pointer-events: auto; position: absolute; }
#transcript { background: #1f2937; border-radius: 8px; max-height: 75vh; overflow: auto; padding: 12px; }
#transcript button { background: none; border: 0; color: inherit; cursor: pointer; display: block; font: inherit; padding: 10px; text-align: left; width: 100%; }
#transcript button.active { background: #164e63; border-radius: 6px; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; } }
</style>
"""

VIEWER_SCRIPT = """
<script>
const video = document.querySelector("video");
const faceLayer = document.querySelector("#faces");
const transcript = document.querySelector("#transcript");
const tracks = new Map(data.tracks.map((track) => [track.track_id, track]));
const transcriptButtons = data.events.map((event) => {
  const button = document.createElement("button");
  button.textContent = `${formatTime(event.start)} — ${event.speaker}: ${event.text}`;
  button.style.color = event.color;
  button.onclick = () => { video.currentTime = event.start; video.play(); };
  transcript.append(button);
  return button;
});

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function boxAt(track, timestamp) {
  if (!track || timestamp < track.start || timestamp > track.end + 0.04) return null;
  let before = track.boxes[0];
  let after = track.boxes[track.boxes.length - 1];
  for (const box of track.boxes) {
    if (box.timestamp <= timestamp) before = box;
    if (box.timestamp >= timestamp) { after = box; break; }
  }
  const span = after.timestamp - before.timestamp;
  const fraction = span ? (timestamp - before.timestamp) / span : 0;
  const interpolate = (first, second) => Object.fromEntries(["x", "y", "width", "height"].map((key) => [key, first[key] + (second[key] - first[key]) * fraction]));
  return { ...interpolate(before, after), bodyBox: interpolate(before.bodyBox, after.bodyBox) };
}

function render() {
  const activeEvents = data.events.filter((event) => event.start <= video.currentTime && video.currentTime < event.end);
  faceLayer.replaceChildren();
  activeEvents.forEach((event) => {
    const box = boxAt(tracks.get(event.faceTrack), video.currentTime);
    if (!box) return;
    const face = document.createElement("button");
    face.className = "speaker";
    face.title = `${event.speaker}: ${event.text}`;
    face.style.setProperty("--speaker-color", event.color);
    Object.assign(face.style, {
      left: `${100 * box.bodyBox.x / data.width}%`, top: `${100 * box.bodyBox.y / data.height}%`,
      width: `${100 * box.bodyBox.width / data.width}%`, height: `${100 * box.bodyBox.height / data.height}%`,
    });
    faceLayer.append(face);
  });
  transcriptButtons.forEach((button, index) => button.classList.toggle("active", activeEvents.includes(data.events[index])));
}

video.addEventListener("timeupdate", render);
video.addEventListener("seeked", render);
render();
</script>
"""


def write_events(path: Path, events: list[CaptionEvent]) -> None:
    """Write final speaker-attribution events as documented JSON.

    Args:
        path: Destination JSON path.
        events: Caption events to serialize.
    """
    path.write_text(
        json.dumps([caption_payload(event) for event in events], indent=2),
        encoding="utf-8",
    )


def write_vtt(path: Path, events: list[CaptionEvent]) -> None:
    """Write speaker-labelled WebVTT captions.

    Args:
        path: Destination VTT path.
        events: Caption events to serialize.
    """
    lines = ["WEBVTT", ""]
    for event in events:
        lines.extend(
            [
                f"{_vtt_timestamp(event.start)} --> {_vtt_timestamp(event.end)}",
                f"{event.speaker}: {event.text}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_viewer(
    path: Path,
    video_name: str,
    tracks: list[FaceTrack],
    events: list[CaptionEvent],
    width: int,
    height: int,
) -> None:
    """Write a standalone result viewer with hoverable active-speaker boxes."""
    active_track_ids = {event.face_track for event in events if event.face_track}
    payload = {
        "width": width,
        "height": height,
        "tracks": [
            track
            for track in track_payloads(tracks, width, height)
            if track["track_id"] in active_track_ids
        ],
        "events": [caption_payload(event) for event in events],
    }
    page = "\n".join(
        (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Active speaker viewer</title>',
            VIEWER_STYLE,
            '<main><h1>Active speaker viewer</h1><section id="player"><video controls src="'
            + video_name
            + '"></video><div id="faces"></div></section><aside id="transcript" aria-label="Transcript"></aside></main>',
            "<script>const data = "
            + json.dumps(payload, ensure_ascii=False)
            + ";</script>",
            VIEWER_SCRIPT,
            "</html>",
        )
    )
    path.write_text(page, encoding="utf-8")


def render_video(
    input_path: Path,
    output_path: Path,
    tracks: list[FaceTrack],
    events: list[CaptionEvent],
    frame_rate: float,
) -> None:
    """Render highlights and burned-in captions, then remux original audio.

    Args:
        input_path: Original video with its audio stream.
        output_path: Final annotated MP4 path.
        tracks: Face rectangles to draw.
        events: Active speaker captions to render.
        frame_rate: Source frame rate for the temporary video stream.

    Raises:
        RuntimeError: If OpenCV or FFmpeg cannot write the final video.
    """
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open input video: {input_path}")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        capture.release()
        raise RuntimeError("input video has no usable dimensions")
    silent_path = output_path.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(
        str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV cannot create the rendered video")
    try:
        timestamps = video_presentation_timestamps(input_path)
        frame_index = 0
        while True:
            success, image = capture.read()
            if not success:
                break
            timestamp = (
                timestamps[frame_index]
                if frame_index < len(timestamps)
                else frame_index / frame_rate
            )
            active_events = _events_at(events, timestamp)
            active_tracks = _active_tracks(active_events)
            for track in tracks:
                event = active_tracks.get(track.track_id)
                if event is None:
                    continue
                box = box_at(track, timestamp)
                if box is None:
                    continue
                _draw_speaker(image, box, event)
            _draw_captions(image, active_events)
            writer.write(image)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    try:
        _remux_audio(silent_path, input_path, output_path)
    finally:
        silent_path.unlink(missing_ok=True)
    LOGGER.info("wrote annotated video %s", output_path)


def _vtt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def _events_at(events: list[CaptionEvent], timestamp: float) -> list[CaptionEvent]:
    return [event for event in events if event.start <= timestamp < event.end]


def _active_tracks(events: list[CaptionEvent]) -> dict[str, CaptionEvent]:
    strongest_events: dict[str, CaptionEvent] = {}
    for event in events:
        if event.face_track is None:
            continue
        existing = strongest_events.get(event.speaker)
        if existing is None or event.confidence > existing.confidence:
            strongest_events[event.speaker] = event
    return {
        event.face_track: event
        for event in strongest_events.values()
        if event.face_track
    }


def track_payloads(
    tracks: list[FaceTrack], width: int, height: int
) -> list[dict[str, object]]:
    """Serialize face tracks with lightweight estimated upper-body rectangles."""
    return [
        track.__dict__
        | {"boxes": [_box_payload(box, width, height) for box in track.boxes]}
        for track in tracks
    ]


def _box_payload(box: object, width: int, height: int) -> dict[str, object]:
    return box.__dict__ | {"bodyBox": body_box(box, width, height).__dict__}


def _draw_speaker(image: object, box: object, event: CaptionEvent) -> None:
    height, width = image.shape[:2]
    estimated_body = body_box(box, width, height)
    color = speaker_color_bgr(event.speaker)
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (estimated_body.x, estimated_body.y),
        (
            estimated_body.x + estimated_body.width,
            estimated_body.y + estimated_body.height,
        ),
        color,
        -1,
    )
    cv2.addWeighted(
        overlay, BODY_HIGHLIGHT_OPACITY, image, 1 - BODY_HIGHLIGHT_OPACITY, 0, image
    )
    cv2.rectangle(
        image,
        (estimated_body.x, estimated_body.y),
        (
            estimated_body.x + estimated_body.width,
            estimated_body.y + estimated_body.height,
        ),
        color,
        4,
    )
    cv2.putText(
        image,
        f"SPEAKING - {event.speaker}",
        (box.x, max(24, box.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def _draw_captions(image: object, events: list[CaptionEvent]) -> None:
    if not events:
        return
    height, width = image.shape[:2]
    maximum_width = width - 2 * CAPTION_HORIZONTAL_MARGIN
    lines = [
        (line, speaker_color_bgr(event.speaker))
        for event in events
        for line in _caption_lines(f"{event.speaker}: {event.text}", maximum_width)
    ]
    top = max(0, height - CAPTION_LINE_HEIGHT * len(lines) - 12)
    cv2.rectangle(image, (0, top), (width, height), CAPTION_BACKGROUND, -1)
    caption_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    caption_drawer = ImageDraw.Draw(caption_image)
    for index, (line, color) in enumerate(lines):
        caption_drawer.text(
            (CAPTION_HORIZONTAL_MARGIN, top + CAPTION_LINE_HEIGHT * index),
            line,
            fill=color[::-1],
            font=_caption_font(),
        )
    image[:] = cv2.cvtColor(np.asarray(caption_image), cv2.COLOR_RGB2BGR)


def _caption_lines(text: str, maximum_width: int) -> list[str]:
    lines: list[str] = []
    current_line = ""
    for word in text.split():
        candidate = f"{current_line} {word}".strip()
        candidate_width = _caption_font().getlength(candidate)
        if current_line and candidate_width > maximum_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = candidate
    return lines + [current_line] if current_line else lines


@lru_cache
def _caption_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(CAPTION_FONT_PATH, CAPTION_FONT_SIZE)


def _remux_audio(silent_path: Path, input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_path),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        message = (
            getattr(error, "stderr", "") or "FFmpeg could not remux the annotated video"
        )
        raise RuntimeError(message.strip()) from error
