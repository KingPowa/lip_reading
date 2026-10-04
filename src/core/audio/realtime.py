"""Incremental local speech decoding, independent of video and attribution."""

import json
from dataclasses import dataclass

import numpy as np

from src.config.live import (
    AUDIO_SAMPLE_RATE,
    MAXIMUM_TRANSCRIPTION_SECONDS,
    PCM_SAMPLE_BYTES,
)
from src.core.contracts import Word


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


class StreamingTranscriber:
    def __init__(self, model: object, origin: float) -> None:
        from vosk import KaldiRecognizer

        self.decoder = KaldiRecognizer(model, AUDIO_SAMPLE_RATE)
        self.decoder.SetWords(True)
        self.origin = origin

    def accept(self, content: bytes, final: bool) -> TranscriptionUpdate:
        results = []
        if content and self.decoder.AcceptWaveform(content):
            results.append(json.loads(self.decoder.Result()))
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
