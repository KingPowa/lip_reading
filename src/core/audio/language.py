"""Language list parsing and spoken-language identification."""

import logging
import re
import threading
from pathlib import Path

import numpy as np

from src.config.live import LANGUAGE_PATTERN, MAXIMUM_SESSION_LANGUAGES
from src.config.streaming import (
    LANGUAGE_ID_DIRECTORY,
    LANGUAGE_ID_MODEL,
    SPEECH_MODELS,
)

LOGGER = logging.getLogger(__name__)


def parse_languages(text: str | None, custom_model: bool = False) -> tuple[str, ...]:
    """Split a comma list into unique, validated languages in priority order."""
    languages = tuple(
        dict.fromkeys(item.strip() for item in (text or "").split(",") if item.strip())
    )
    for language in languages:
        if not re.fullmatch(LANGUAGE_PATTERN, language):
            raise ValueError("invalid transcription language")
        if not custom_model and language not in SPEECH_MODELS:
            raise ValueError(
                f"choose supported speech languages: {', '.join(SPEECH_MODELS)}"
            )
    if len(languages) > MAXIMUM_SESSION_LANGUAGES:
        raise ValueError(f"choose at most {MAXIMUM_SESSION_LANGUAGES} languages")
    if custom_model and len(languages) > 1:
        raise ValueError("--speech-model supports exactly one language")
    return languages


class LanguageIdentifier:
    """Spoken-language probabilities from a small CPU faster-whisper model."""

    def __init__(self, model_directory: Path) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            LANGUAGE_ID_MODEL,
            device="cpu",
            compute_type="int8",
            download_root=str(model_directory / LANGUAGE_ID_DIRECTORY),
        )
        self.lock = threading.Lock()

    def probabilities(self, samples: np.ndarray) -> dict[str, float]:
        with self.lock:
            _, _, scores = self.model.detect_language(samples)
        return {language: float(score) for language, score in scores}


def load_language_identifier(model_directory: Path) -> LanguageIdentifier | None:
    try:
        return LanguageIdentifier(model_directory)
    except Exception as error:  # noqa: BLE001 - captions must not depend on it
        LOGGER.warning(
            "language identification unavailable; using recognizer confidence: %s",
            error,
        )
        return None
