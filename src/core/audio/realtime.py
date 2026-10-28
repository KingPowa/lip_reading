"""Incremental local speech decoding, independent of video and attribution."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from src.config.live import (
    AUDIO_SAMPLE_RATE,
    LANGUAGE_ONE_OFF_PROBABILITY,
    LANGUAGE_ONE_OFF_SECONDS,
    LANGUAGE_SWITCH_PROBABILITY,
    MAXIMUM_TRANSCRIPTION_SECONDS,
    MAXIMUM_UTTERANCE_SECONDS,
    MINIMUM_LANGUAGE_ID_SECONDS,
    PAUSE_CUT_HARD_UTTERANCE_SECONDS,
    PAUSE_CUT_MINIMUM_UTTERANCE_SECONDS,
    PAUSE_CUT_SILENCE_SECONDS,
    PAUSE_CUT_SOFT_UTTERANCE_SECONDS,
    PCM_SAMPLE_BYTES,
)
from src.core.audio.activity import SpeechActivityGate
from src.core.contracts import Word

LOGGER = logging.getLogger(__name__)


def decode_pcm(
    content: bytes, maximum_seconds: float = MAXIMUM_TRANSCRIPTION_SECONDS
) -> np.ndarray:
    maximum_bytes = round(maximum_seconds * AUDIO_SAMPLE_RATE) * PCM_SAMPLE_BYTES
    if not content or len(content) % PCM_SAMPLE_BYTES or len(content) > maximum_bytes:
        raise ValueError(
            "audio must be bounded mono 16 kHz little-endian signed 16-bit PCM"
        )
    return (
        np.frombuffer(content, dtype="<i2").astype(np.float32) / np.iinfo(np.int16).max
    )


@dataclass(frozen=True)
class TranscriptionUpdate:
    words: list[Word]
    partial_text: str
    language: str | None = None


class PauseCutter:
    """Tell when to force an utterance end because Vosk's endpointer waits too long."""

    def __init__(self) -> None:
        self.gate = SpeechActivityGate()
        self.reset()

    def reset(self) -> None:
        self.length = 0
        self.silence = 0
        self.dip = False

    def feed(self, content: bytes) -> bool:
        frame = AUDIO_SAMPLE_RATE // 10 * PCM_SAMPLE_BYTES
        for start in range(0, len(content), frame):
            samples = decode_pcm(content[start : start + frame], 1.0)
            # Dip is judged against the threshold before the gate adapts to the frame.
            energy = float(np.sqrt(np.mean(samples * samples)))
            self.dip = energy < self.gate._threshold()
            speaking = self.gate.update(samples)
            self.silence = 0 if speaking else self.silence + len(samples)
            # Leading silence does not count, so idle audio never forces cuts.
            if speaking or self.length:
                self.length += len(samples)
        length, silence = (
            self.length / AUDIO_SAMPLE_RATE,
            self.silence / AUDIO_SAMPLE_RATE,
        )  # samples are counted as integers so the caps do not drift
        return (
            length >= PAUSE_CUT_HARD_UTTERANCE_SECONDS
            or (length >= PAUSE_CUT_SOFT_UTTERANCE_SECONDS and self.dip)
            or (
                length >= PAUSE_CUT_MINIMUM_UTTERANCE_SECONDS
                and silence >= PAUSE_CUT_SILENCE_SECONDS
            )
        )


class StreamingTranscriber:
    def __init__(self, model: object, origin: float) -> None:
        from vosk import KaldiRecognizer

        self.decoder = KaldiRecognizer(model, AUDIO_SAMPLE_RATE)
        self.decoder.SetWords(True)
        self.origin = origin
        self.pauses = PauseCutter()

    def accept(self, content: bytes, final: bool) -> TranscriptionUpdate:
        results = []
        ended = cut = False
        if content:
            ended = self.decoder.AcceptWaveform(content)
            cut = self.pauses.feed(content)
            if ended:
                results.append(json.loads(self.decoder.Result()))
        if ended or cut:
            self.pauses.reset()
        if cut and not (ended or final):
            results.append(json.loads(self.decoder.FinalResult()))
        if final:
            results.append(json.loads(self.decoder.FinalResult()))
        words = [
            Word(
                item["word"],
                self.origin + float(item["start"]),
                self.origin + float(item["end"]),
                float(item["conf"]),
            )
            for result in results
            for item in result.get("result", [])
        ]
        partial = (
            "" if final else json.loads(self.decoder.PartialResult()).get("partial", "")
        )
        return TranscriptionUpdate(words, partial)


def _words(results: list[dict], origin: float) -> list[Word]:
    return [
        Word(
            item["word"],
            origin + float(item["start"]),
            origin + float(item["end"]),
            float(item["conf"]),
        )
        for result in results
        for item in result.get("result", [])
    ]


class MultilingualTranscriber:
    """Decode with one active language and switch only when two consecutive utterances agree.

    ``identify`` maps utterance samples to language probabilities. Vosk word times
    count every sample ever fed to a recognizer (``Reset`` keeps the clock), so
    ``clock`` and ``shift`` map each recognizer's times onto the stream.
    """

    def __init__(
        self,
        models: dict[str, object],
        origin: float,
        identify: Callable[[np.ndarray], dict[str, float]] | None = None,
    ) -> None:
        from vosk import KaldiRecognizer

        self.decoders = {}
        for language, model in models.items():
            self.decoders[language] = KaldiRecognizer(model, AUDIO_SAMPLE_RATE)
            self.decoders[language].SetWords(True)
        self.language = next(iter(models))
        self.origin = origin
        self.identify = identify
        self.utterance = bytearray()
        self.fed = 0.0
        self.clock = dict.fromkeys(models, 0.0)
        self.shift = 0.0
        # Utterance that voted for another language, withheld until the next one decides.
        self.held: tuple[str, float, bytes, list[Word]] | None = None
        self.pauses = PauseCutter()

    def accept(self, content: bytes, final: bool) -> TranscriptionUpdate:
        decoder = self.decoders[self.language]
        ended = cut = False
        if content:
            self._remember(content)
            ended = decoder.AcceptWaveform(content)
            cut = self.pauses.feed(content)
        if ended or cut:
            self.pauses.reset()
        if not (ended or cut or final):
            partial = json.loads(decoder.PartialResult())
            return TranscriptionUpdate([], partial.get("partial", ""), self.language)
        results = [json.loads(decoder.Result())] if ended else []
        if final or (cut and not ended):
            results.append(json.loads(decoder.FinalResult()))
        words = _words(results, self.origin + self.shift)
        samples = bytes(self.utterance)
        self.utterance.clear()
        target, probability = self._vote(samples)
        released: list[Word] = []
        if self.held:
            language, held_probability, held_samples, held_words = self.held
            self.held = None
            if target == language and len(held_samples) + len(samples) <= self._maximum():
                words = self._redecode(target, held_samples + samples)
                target = None
            else:  # refuted, unconfirmed or too long to decode together: release it
                released = held_words
                if (
                    held_probability >= LANGUAGE_ONE_OFF_PROBABILITY
                    and self._seconds(held_samples) >= LANGUAGE_ONE_OFF_SECONDS
                ):  # a clearly foreign sentence: decode it alone, keep the language
                    released = self._redecode(
                        language, held_samples, self._seconds(samples), switch=False
                    )
        if target and not final:  # a third language is held in turn
            self.held = (target, probability, samples, words)
            words = []
        words = released + words
        decoder = self.decoders[self.language]
        partial = (
            "" if final else json.loads(decoder.PartialResult()).get("partial", "")
        )
        return TranscriptionUpdate(
            [word if word.language else replace(word, language=self.language) for word in words],
            partial,
            self.language,
        )

    def _remember(self, content: bytes) -> None:
        self.utterance += content
        seconds = len(content) / PCM_SAMPLE_BYTES / AUDIO_SAMPLE_RATE
        self.fed += seconds
        self.clock[self.language] += seconds
        del self.utterance[: -self._maximum()]

    @staticmethod
    def _maximum() -> int:
        return round(MAXIMUM_UTTERANCE_SECONDS * AUDIO_SAMPLE_RATE) * PCM_SAMPLE_BYTES

    @staticmethod
    def _seconds(samples: bytes) -> float:
        return len(samples) / PCM_SAMPLE_BYTES / AUDIO_SAMPLE_RATE

    def _vote(self, samples: bytes) -> tuple[str | None, float]:
        """Return the other language this utterance points to, with its probability."""
        probabilities = self._probabilities(samples)
        if not probabilities:
            return None, 0.0
        best = max(probabilities, key=probabilities.get)
        if best == self.language or probabilities[best] < LANGUAGE_SWITCH_PROBABILITY:
            return None, 0.0
        return best, probabilities[best]

    def _redecode(
        self, language: str, samples: bytes, tail: float = 0.0, switch: bool = True
    ) -> list[Word]:
        """Decode buffered audio again with another recognizer; ``tail`` is the audio fed after it.

        Without ``switch`` the words come back tagged with ``language`` and the active
        language and shift stay as they are.
        """
        decoder = self.decoders[language]
        decoder.Reset()
        seconds = self._seconds(samples)
        shift = self.fed - tail - seconds - self.clock[language]
        self.clock[language] += seconds
        if switch:
            self.language, self.shift = language, shift
        results = []
        if decoder.AcceptWaveform(samples):
            results.append(json.loads(decoder.Result()))
        results.append(json.loads(decoder.FinalResult()))
        words = _words(results, self.origin + shift)
        return words if switch else [replace(word, language=language) for word in words]

    def _probabilities(self, content: bytes) -> dict[str, float]:
        seconds = len(content) / PCM_SAMPLE_BYTES / AUDIO_SAMPLE_RATE
        if self.identify is None or seconds < MINIMUM_LANGUAGE_ID_SECONDS:
            return {}
        try:
            scores = self.identify(decode_pcm(content, MAXIMUM_UTTERANCE_SECONDS))
        except Exception as error:  # noqa: BLE001 - keep the current language
            LOGGER.warning("language identification failed: %s", error)
            return {}
        allowed = {language: scores.get(language, 0.0) for language in self.decoders}
        total = sum(allowed.values())
        return (
            {language: score / total for language, score in allowed.items()}
            if total
            else {}
        )
