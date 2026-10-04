"""Command-line orchestration for offline active-speaker captioning."""

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from src.app.rendering.offline import (
    render_video,
    track_payloads,
    write_events,
    write_viewer,
    write_vtt,
)
from src.config.models import (
    FACE_LANDMARKER_FILENAME,
    LIGHT_ASD_FILENAME,
    MODEL_DIRECTORY,
    WHISPER_MODEL_NAME,
)
from src.config.vision import (
    MINIMUM_LIVE_FACE_DETECTIONS,
    MINIMUM_LIVE_TRACK_DURATION_SECONDS,
    MINIMUM_TRACK_DETECTIONS,
    MINIMUM_TRACK_DURATION_SECONDS,
)
from src.core.audio.speech import (
    diarize_turns,
    transcribe_speech,
    uncommitted_words,
    write_speech_json,
)
from src.core.contracts import CaptionEvent, FaceTrack, SpeakerTurn, Word
from src.core.media import (
    MediaError,
    MediaInfo,
    extract_audio,
    inspect_media,
    read_video_frames,
)
from src.core.tracking.association import associate_speakers
from src.core.vision.faces import analyze_faces
from src.core.vision.light_asd import score_face_activity

LOGGER = logging.getLogger(__name__)
from src.config.offline import LIVE_ACTIVITY_CONTEXT_SECONDS, UNASSIGNED_LIVE_VOICE


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for one prerecorded video.

    Returns:
        The configured command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Caption and highlight speakers in a prerecorded video"
    )
    parser.add_argument(
        "input", type=Path, help="MP4 or other FFmpeg-supported video with audio"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for annotated artifacts",
    )
    parser.add_argument(
        "--whisper-model", default=WHISPER_MODEL_NAME, help="Local Whisper model name"
    )
    parser.add_argument("--language", help="Optional ISO language code for Whisper")
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=MODEL_DIRECTORY,
        help="Local model download directory",
    )
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
    write_viewer(
        output_paths["viewer"],
        output_paths["video"].name,
        tracks,
        events,
        info.width,
        info.height,
    )
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
    initial_prompt: str | None = None,
    minimum_word_timestamp: float = float("-inf"),
    maximum_word_timestamp: float = float("inf"),
    timings: dict[str, float] | None = None,
    activity_device: str = "cpu",
) -> tuple[
    MediaInfo, list[FaceTrack], list[CaptionEvent], list[SpeakerTurn], list[Word]
]:
    """Run the shared recognition pipeline without producing rendered artifacts."""
    LOGGER.info("inspecting input %s", input_path)
    timings = timings if timings is not None else {}
    stage_started = time.perf_counter()
    info = inspect_media(input_path)
    frames = read_video_frames(input_path)
    timings["decode"] = time.perf_counter() - stage_started
    LOGGER.info("decoded %d frames over %.2f seconds", len(frames), info.duration)
    stage_started = time.perf_counter()
    turns, words = transcribe_speech(
        audio_path,
        whisper_model,
        model_directory / "whisper",
        language,
        transcription_model,
        True,
        initial_prompt,
    )
    timings["transcription"] = time.perf_counter() - stage_started
    caption_words = uncommitted_words(
        words, initial_prompt, minimum_word_timestamp, maximum_word_timestamp
    )
    stage_started = time.perf_counter()
    if not live:
        turns = diarize_turns(audio_path, turns, model_directory, speaker_centroids)
    timings["diarization"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    tracks = analyze_faces(
        frames,
        model_directory / FACE_LANDMARKER_FILENAME,
        MINIMUM_LIVE_FACE_DETECTIONS if live else MINIMUM_TRACK_DETECTIONS,
        MINIMUM_LIVE_TRACK_DURATION_SECONDS if live else MINIMUM_TRACK_DURATION_SECONDS,
        use_face_detector=live,
    )
    if track_id_transformer is not None:
        track_id_transformer(tracks)
    timings["faces"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    activity_start = float("-inf")
    activity_end = float("inf")
    if live:
        activity_start = (
            min((word.start for word in caption_words), default=float("inf"))
            - LIVE_ACTIVITY_CONTEXT_SECONDS
        )
        activity_end = (
            max((word.end for word in caption_words), default=float("-inf"))
            + LIVE_ACTIVITY_CONTEXT_SECONDS
        )
    activities = []
    if not live or caption_words:
        activities = score_face_activity(
            audio_path,
            frames,
            tracks,
            model_directory / LIGHT_ASD_FILENAME,
            activity_device,
            activity_start,
            activity_end,
        )
    timings["activity"] = time.perf_counter() - stage_started
    if live and speaker_centroids is not None:
        stage_started = time.perf_counter()
        visual_events = associate_speakers(
            _live_word_turns(caption_words, turns),
            caption_words,
            tracks,
            activities,
            prefer_track_labels=True,
            merge_words=True,
        )
        visual_turns = [
            SpeakerTurn(
                event.face_track or f"{UNASSIGNED_LIVE_VOICE}-{event.state}-{index}",
                event.start,
                event.end,
                event.confidence,
            )
            for index, event in enumerate(visual_events)
        ]
        turns = diarize_turns(
            audio_path,
            visual_turns,
            model_directory,
            speaker_centroids,
            preserve_short_turn_ids=True,
        )
        timings["diarization"] = time.perf_counter() - stage_started
    caption_turns = turns
    if live:
        caption_turns = _live_word_turns(caption_words, turns)
    events = associate_speakers(
        caption_turns,
        caption_words,
        tracks,
        activities,
        voice_labels,
        track_labels,
        prefer_track_labels=live and speaker_centroids is None,
        merge_words=live,
    )
    return info, tracks, events, turns, words


def _live_word_turns(words: list[Word], turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    word_turns = []
    for word in words:
        if word.end <= word.start:
            continue
        midpoint = (word.start + word.end) / 2
        speaker_id = next(
            (turn.speaker_id for turn in turns if turn.start <= midpoint <= turn.end),
            UNASSIGNED_LIVE_VOICE,
        )
        word_turns.append(
            SpeakerTurn(speaker_id, word.start, word.end, word.confidence)
        )
    return word_turns


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
