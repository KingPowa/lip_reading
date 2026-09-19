"""Local transcription and anonymous voice clustering."""

import json
import logging
import wave
from collections.abc import Iterable
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


def transcribe_speech(
    audio_path: Path,
    model_name: str,
    model_directory: Path,
    language: str | None = None,
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
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError("Whisper is not installed") from error

    model_directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("loading Whisper model %s", model_name)
    model = whisper.load_model(model_name, download_root=str(model_directory))
    LOGGER.info("transcribing audio with Whisper")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language,
        fp16=False,
        verbose=False,
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


def diarize_turns(audio_path: Path, turns: list[SpeakerTurn], model_directory: Path) -> list[SpeakerTurn]:
    """Assign stable anonymous speaker IDs to Whisper turns using ECAPA embeddings.

    Args:
        audio_path: Mono 16 kHz WAV audio used to produce the turns.
        turns: Whisper speech segments in presentation-timestamp order.
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
    centroids: list[np.ndarray] = []
    labeled_turns: list[SpeakerTurn] = []
    previous_speaker_id = FIRST_SPEAKER_ID
    for turn in turns:
        embedding = _turn_embedding(classifier, samples, sample_rate, turn)
        speaker_id = previous_speaker_id if embedding is None else _cluster_embedding(embedding, centroids)
        labeled_turns.append(SpeakerTurn(speaker_id, turn.start, turn.end, turn.confidence))
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
    confidence = float(np.exp(float(segment.get("avg_logprob", np.log(DEFAULT_CONFIDENCE)))))
    return SpeakerTurn(
        FIRST_SPEAKER_ID,
        float(segment["start"]),
        float(segment["end"]),
        float(np.clip(confidence, 0.0, 1.0)),
    )


def _audio_samples(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path)) as audio:
        sample_rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16).astype(np.float32)
    return samples / np.iinfo(np.int16).max, sample_rate


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


def _turn_embedding(classifier: object, samples: np.ndarray, sample_rate: int, turn: SpeakerTurn) -> np.ndarray | None:
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
