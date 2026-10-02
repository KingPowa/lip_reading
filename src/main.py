"""Command-line orchestration for offline active-speaker captioning."""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from .association import associate_speakers
from .config import (
    FACE_LANDMARKER_FILENAME,
    LIGHT_ASD_FILENAME,
    MAXIMUM_ACTIVITY_TRACKS,
    MAXIMUM_LIVE_ACTIVITY_TRACKS,
    MINIMUM_LIVE_FACE_DETECTIONS,
    MINIMUM_LIVE_TRACK_DURATION_SECONDS,
    MINIMUM_TRACK_DETECTIONS,
    MINIMUM_TRACK_DURATION_SECONDS,
    MODEL_DIRECTORY,
    WHISPER_MODEL_NAME,
)
from .light_asd import score_face_activity
from .contracts import CaptionEvent, FaceTrack, SpeakerTurn, Word
from .media import MediaError, MediaInfo, extract_audio, inspect_media, read_video_frames
from .render import render_video, track_payloads, write_events, write_viewer, write_vtt
from .speech import diarize_turns, transcribe_speech, write_speech_json
from .vision import analyze_faces


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for one prerecorded video.

    Returns:
        The configured command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Caption and highlight speakers in a prerecorded video")
    parser.add_argument("input", type=Path, help="MP4 or other FFmpeg-supported video with audio")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Directory for annotated artifacts")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL_NAME, help="Local Whisper model name")
    parser.add_argument("--language", help="Optional ISO language code for Whisper")
    parser.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY, help="Local model download directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def run(
    input_path: Path,
    output_dir: Path,
    whisper_model: str = WHISPER_MODEL_NAME,
    language: str | None = None,
    model_directory: Path = MODEL_DIRECTORY,
) -> dict[str, Path]:
    """Create an annotated video, WebVTT captions, and attribution JSON.

    Args:
        input_path: Source video containing one video and one audio stream.
        output_dir: Directory for final and diagnostic artifacts.
        whisper_model: Whisper model to load for local transcription.
        language: Optional ISO language code passed to Whisper.
        model_directory: Directory containing or receiving pretrained model files.

    Returns:
        Named paths for the final video, captions, events, and diagnostics.

    Raises:
        MediaError: If the source does not contain usable video and audio.
        RuntimeError: If local model inference or rendering fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = extract_audio(input_path, output_dir / "audio.wav")
    info, tracks, events, turns, words = recognize(
        input_path, audio_path, whisper_model, language, model_directory
    )
    output_paths = {
        "video": output_dir / "output.annotated.mp4",
        "captions": output_dir / "captions.vtt",
        "events": output_dir / "events.json",
        "speech": output_dir / "speech.json",
        "faces": output_dir / "faces.json",
        "viewer": output_dir / "viewer.html",
    }
    write_speech_json(output_paths["speech"], turns, words)
    output_paths["faces"].write_text(
        json.dumps(track_payloads(tracks, info.width, info.height), indent=2),
        encoding="utf-8",
    )
    write_events(output_paths["events"], events)
    write_vtt(output_paths["captions"], events)
    render_video(input_path, output_paths["video"], tracks, events, info.frame_rate)
    write_viewer(output_paths["viewer"], output_paths["video"].name, tracks, events, info.width, info.height)
    LOGGER.info("completed captioning with %d event(s)", len(events))
    return output_paths


def recognize(
    input_path: Path,
    audio_path: Path,
    whisper_model: str = WHISPER_MODEL_NAME,
    language: str | None = None,
    model_directory: Path = MODEL_DIRECTORY,
    transcription_model: object | None = None,
    speaker_centroids: object | None = None,
    voice_labels: dict[str, str] | None = None,
    track_labels: dict[str, str] | None = None,
    live: bool = False,
    track_id_transformer: Callable[[list[FaceTrack]], None] | None = None,
) -> tuple[MediaInfo, list[FaceTrack], list[CaptionEvent], list[SpeakerTurn], list[Word]]:
    """Run the shared recognition pipeline without producing rendered artifacts."""
    LOGGER.info("inspecting input %s", input_path)
    info = inspect_media(input_path)
    frames = read_video_frames(input_path)
    LOGGER.info("decoded %d frames over %.2f seconds", len(frames), info.duration)
    turns, words = transcribe_speech(
        audio_path,
        whisper_model,
        model_directory / "whisper",
        language,
        transcription_model,
        not live,
    )
    if not live:
        turns = diarize_turns(audio_path, turns, model_directory, speaker_centroids)
    tracks = analyze_faces(
        frames,
        model_directory / FACE_LANDMARKER_FILENAME,
        MINIMUM_LIVE_FACE_DETECTIONS if live else MINIMUM_TRACK_DETECTIONS,
        MINIMUM_LIVE_TRACK_DURATION_SECONDS if live else MINIMUM_TRACK_DURATION_SECONDS,
    )
    if track_id_transformer is not None:
        track_id_transformer(tracks)
    activities = score_face_activity(
        audio_path,
        frames,
        tracks,
        model_directory / LIGHT_ASD_FILENAME,
        MAXIMUM_LIVE_ACTIVITY_TRACKS if live else MAXIMUM_ACTIVITY_TRACKS,
    )
    events = associate_speakers(
        turns,
        words,
        tracks,
        activities,
        voice_labels,
        track_labels,
        live,
    )
    return info, tracks, events, turns, words


def main() -> None:
    """Parse arguments, configure logs, and run the captioning pipeline."""
    parser = build_parser()
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        paths = run(
            arguments.input,
            arguments.output,
            arguments.whisper_model,
            arguments.language,
            arguments.model_directory,
        )
    except (MediaError, RuntimeError, ValueError) as error:
        LOGGER.exception("captioning failed")
        parser.error(str(error))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
