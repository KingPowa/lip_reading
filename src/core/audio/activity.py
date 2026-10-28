"""Adaptive speech detection with an estimated noise floor and hysteresis."""

import numpy as np

from src.config.live import (
    INITIAL_NOISE_FLOOR,
    MAXIMUM_NOISE_FLOOR,
    MINIMUM_NOISE_FLOOR,
    MINIMUM_SPEECH_ENERGY,
    NOISE_FLOOR_FALL,
    NOISE_FLOOR_RISE_SILENT,
    NOISE_FLOOR_RISE_SPEAKING,
    SPEECH_OFF_ABSOLUTE_RATIO,
    SPEECH_OFF_NOISE_RATIO,
    SPEECH_ON_NOISE_RATIO,
)


class SpeechActivityGate:
    """Decide per packet whether audio is speech relative to the room noise."""

    def __init__(self) -> None:
        self.floor = INITIAL_NOISE_FLOOR
        self.speaking = False

    def update(self, samples: np.ndarray) -> bool:
        energy = float(np.sqrt(np.mean(samples * samples)))
        self.speaking = energy >= self._threshold()
        self._adapt(energy)
        return self.speaking

    def _threshold(self) -> float:
        if self.speaking:
            return max(
                MINIMUM_SPEECH_ENERGY * SPEECH_OFF_ABSOLUTE_RATIO,
                SPEECH_OFF_NOISE_RATIO * self.floor,
            )
        return max(MINIMUM_SPEECH_ENERGY, SPEECH_ON_NOISE_RATIO * self.floor)

    def _adapt(self, energy: float) -> None:
        if energy < self.floor:
            rate = NOISE_FLOOR_FALL
        elif self.speaking:
            rate = NOISE_FLOOR_RISE_SPEAKING
        else:
            rate = NOISE_FLOOR_RISE_SILENT
        floor = self.floor + rate * (energy - self.floor)
        self.floor = min(MAXIMUM_NOISE_FLOOR, max(MINIMUM_NOISE_FLOOR, floor))
