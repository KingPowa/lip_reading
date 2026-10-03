"""Local transcription and anonymous voice clustering."""

import json
import logging
import re
import wave
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import (
    MINIMUM_VOICE_SEGMENT_SECONDS,
    SPEAKER_MODEL_DIRECTORY,
    SPEAKER_MODEL_SOURCE,
    VOICE_CLUSTER_THRESHOLD,
)
from .contracts import SpeakerTurn, Word

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIDENCE = 0.5
FIRST_SPEAKER_ID = "SPEAKER_00"
MAXIMUM_OVERLAP_WORDS = 32
MINIMUM_ANCHOR_WORDS = 2
ANCHOR_TIMESTAMP_TOLERANCE_SECONDS = 1.5
NORMALIZED_AUDIO_SAMPLE_RATE = 16000
NORMALIZED_AUDIO_CHANNELS = 1
NORMALIZED_AUDIO_SAMPLE_WIDTH_BYTES = 2


def transcribe_speech(
    audio_path: Path,
    model_name: str,
    model_directory: Path,
    language: str | None = None,
    model: object | None = None,
    word_timestamps: bool = True,
    initial_prompt: str | None = None,
) -> tuple[list[SpeakerTurn], list[Word]]:
    """Transcribe normalized audio with local Whisper word timestamps.

    Args:
        audio_path: Mono 16 kHz WAV audio extracted from the input video.
        model_name: Whisper model name, such as ``base`` or ``small``.
        model_directory: Directory holding downloaded Whisper weights.
        language: Optional ISO language code; automatic detection is used when absent.

    Returns:
        Whisper segments as anonymous turns and their timestamped words.

    Raises:
        RuntimeError: If Whisper cannot load or transcribe the audio.
    """
    model = model or load_transcription_model(model_name, model_directory)
    samples, _ = _audio_samples(audio_path)
    LOGGER.info("transcribing audio with Whisper")
    result = model.transcribe(
        samples,
        word_timestamps=word_timestamps,
        language=language.lower() if language else None,
        task="transcribe",
        fp16=False,
        verbose=False,
        initial_prompt=initial_prompt,
    )
    turns = [_turn_from_segment(segment) for segment in result.get("segments", [])]
    words = [
        Word(
            item["word"].strip(),
            float(item["start"]),
            float(item["end"]),
            float(item.get("probability", DEFAULT_CONFIDENCE)),
        )
        for segment in result.get("segments", [])
        for item in segment.get("words", [])
        if item["word"].strip()
    ]
    if not words:
        words = [
            Word(str(segment["text"]).strip(), turn.start, turn.end, turn.confidence)
            for segment, turn in zip(result.get("segments", []), turns, strict=True)
            if str(segment["text"]).strip()
        ]
    LOGGER.info("Whisper produced %d turns and %d words", len(turns), len(words))
    return turns, words


def load_transcription_model(model_name: str, model_directory: Path) -> object:
    """Load one local Whisper model for reusable batch or stream transcription."""
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError("Whisper is not installed") from error

    model_directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("loading Whisper model %s", model_name)
    return whisper.load_model(model_name, download_root=str(model_directory))


def load_faster_whisper_model(model_name: str) -> object:
    """Load the CPU INT8 faster-whisper model used by live recognition."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("faster-whisper is not installed") from error

    repository = f"Systran/faster-whisper-{model_name}"
    LOGGER.info("loading faster-whisper model %s on CPU (INT8)", repository)
    return FasterWhisperAdapter(
        WhisperModel(repository, device="cpu", compute_type="int8", cpu_threads=4)
    )


class FasterWhisperAdapter:
    """Adapt faster-whisper's lazy segment iterator to the shared speech path."""

    def __init__(self, model: object) -> None:
        self.model = model

    def transcribe(self, audio: np.ndarray, **options: object) -> dict[str, list[dict]]:
        # Prior text anchors overlap; prompting partial windows replayed earlier phrases.
        segments, _ = self.model.transcribe(
            audio,
            language=options.get("language"),
            task="transcribe",
            word_timestamps=options.get("word_timestamps", True),
            condition_on_previous_text=False,
            beam_size=5,
        )
        return {
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                            if word.probability is not None
                            else DEFAULT_CONFIDENCE,
                        }
                        for word in (segment.words or [])
                    ],
                }
                for segment in segments
            ]
        }


def uncommitted_words(
    words: list[Word],
    previous_text: str | None,
    minimum_timestamp: float,
    maximum_timestamp: float,
) -> list[Word]:
    """Use an overlap text anchor when Whisper revises committed word timestamps."""

    # ponytail: exact suffix anchors; use a streaming aligner if ASR rewrites whole phrases.
    def normalize_word(text: str) -> str:
        return re.sub(r"[^\w]", "", text.casefold())

    previous = [normalize_word(word) for word in (previous_text or "").split()][
        -MAXIMUM_OVERLAP_WORDS:
    ]
    current = [normalize_word(word.text) for word in words]
    matches = SequenceMatcher(
        None, previous, current, autojunk=False
    ).get_matching_blocks()
    anchors = [
        match.b + match.size
        for match in matches
        if match.size >= MINIMUM_ANCHOR_WORDS
        and match.a + match.size == len(previous)
        and words[match.b].start <= minimum_timestamp
        and abs(words[match.b + match.size - 1].end - minimum_timestamp)
        <= ANCHOR_TIMESTAMP_TOLERANCE_SECONDS
    ]
    if anchors:
        boundary = min(
            anchors, key=lambda index: abs(words[index - 1].end - minimum_timestamp)
        )
        return [word for word in words[boundary:] if word.end <= maximum_timestamp]
    return [
        word
        for word in words
        if (word.start + word.end) / 2 > minimum_timestamp
        and word.end <= maximum_timestamp
    ]


def diarize_turns(
    audio_path: Path,
    turns: list[SpeakerTurn],
    model_directory: Path,
    centroids: list[np.ndarray] | None = None,
    preserve_short_turn_ids: bool = False,
) -> list[SpeakerTurn]:
    """Assign stable anonymous speaker IDs to speech intervals using ECAPA embeddings.

    Args:
        audio_path: Mono 16 kHz WAV audio used to produce the turns.
        turns: Speech intervals in presentation-timestamp order.
        model_directory: Directory holding the public ECAPA model files.

    Returns:
        The input turns with greedily clustered ``SPEAKER_nn`` identities.

    Raises:
        RuntimeError: If the speaker embedding model cannot be loaded.
    """
    if not turns:
        return []
    samples, sample_rate = _audio_samples(audio_path)
    classifier = _speaker_classifier(model_directory / SPEAKER_MODEL_DIRECTORY)
    centroids = centroids if centroids is not None else []
    labeled_turns: list[SpeakerTurn] = []
    previous_speaker_id = FIRST_SPEAKER_ID
    for turn in turns:
        embedding = _turn_embedding(classifier, samples, sample_rate, turn)
        speaker_id = turn.speaker_id if preserve_short_turn_ids else previous_speaker_id
        if embedding is not None:
            speaker_id = _cluster_embedding(embedding, centroids)
        labeled_turns.append(
            SpeakerTurn(speaker_id, turn.start, turn.end, turn.confidence)
        )
        previous_speaker_id = speaker_id
    LOGGER.info("ECAPA clustered %d turns into %d voice(s)", len(turns), len(centroids))
    return labeled_turns


def write_speech_json(path: Path, turns: list[SpeakerTurn], words: list[Word]) -> None:
    """Write normalized transcription and diarization output for inspection.

    Args:
        path: Destination JSON path.
        turns: Voice-labelled speech intervals.
        words: Whisper words on the same presentation timeline.
    """
    payload = {
        "turns": [turn.__dict__ for turn in turns],
        "words": [word.__dict__ for word in words],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _turn_from_segment(segment: dict[str, object]) -> SpeakerTurn:
    confidence = float(
        np.exp(float(segment.get("avg_logprob", np.log(DEFAULT_CONFIDENCE))))
    )
    return SpeakerTurn(
        FIRST_SPEAKER_ID,
        float(segment["start"]),
        float(segment["end"]),
        float(np.clip(confidence, 0.0, 1.0)),
    )


def _audio_samples(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path)) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
            NORMALIZED_AUDIO_SAMPLE_RATE,
            NORMALIZED_AUDIO_CHANNELS,
            NORMALIZED_AUDIO_SAMPLE_WIDTH_BYTES,
        ):
            raise RuntimeError("speech models require 16 kHz mono 16-bit PCM audio")
        sample_rate = audio.getframerate()
        samples = np.frombuffer(
            audio.readframes(audio.getnframes()), dtype=np.int16
        ).astype(np.float32)
    return samples / np.iinfo(np.int16).max, sample_rate


def warm_up_speaker_model(model_directory: Path) -> None:
    """Load the cached voice embedding model before accepting live speech."""
    _speaker_classifier(model_directory / SPEAKER_MODEL_DIRECTORY)


@lru_cache
def _speaker_classifier(model_directory: Path):
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError as error:
        raise RuntimeError("SpeechBrain is not installed") from error

    model_directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("loading ECAPA speaker embedding model")
    return EncoderClassifier.from_hparams(
        source=SPEAKER_MODEL_SOURCE,
        savedir=str(model_directory),
        run_opts={"device": "cpu"},
    )


def _turn_embedding(
    classifier: object, samples: np.ndarray, sample_rate: int, turn: SpeakerTurn
) -> np.ndarray | None:
    import torch

    start = max(0, int(turn.start * sample_rate))
    end = min(len(samples), int(turn.end * sample_rate))
    if end - start < int(MINIMUM_VOICE_SEGMENT_SECONDS * sample_rate):
        return None
    signal = torch.from_numpy(samples[start:end]).unsqueeze(0)
    with torch.inference_mode():
        embedding = classifier.encode_batch(signal).squeeze().cpu().numpy()
    return embedding / np.linalg.norm(embedding)


def _cluster_embedding(embedding: np.ndarray, centroids: list[np.ndarray]) -> str:
    similarities = [float(np.dot(embedding, centroid)) for centroid in centroids]
    best_index = int(np.argmax(similarities)) if similarities else -1
    if best_index < 0 or similarities[best_index] < VOICE_CLUSTER_THRESHOLD:
        centroids.append(embedding)
        return f"SPEAKER_{len(centroids) - 1:02}"
    centroids[best_index] = _normalized_mean((centroids[best_index], embedding))
    return f"SPEAKER_{best_index:02}"


def _normalized_mean(vectors: Iterable[np.ndarray]) -> np.ndarray:
    mean = np.mean(list(vectors), axis=0)
    return mean / np.linalg.norm(mean)
