import unittest

import numpy as np

from src.core.audio.activity import SpeechActivityGate

QUIET_FLOOR = 0.0008


def packet(level):
    return np.full(1600, level, dtype=np.float32)


class SpeechActivityGateTest(unittest.TestCase):
    def settled(self, level=QUIET_FLOOR):
        gate = SpeechActivityGate()
        for _ in range(20):
            gate.update(packet(level))
        return gate

    def test_quiet_speech_over_a_quiet_room_is_detected(self):
        gate = self.settled()
        self.assertFalse(gate.update(packet(QUIET_FLOOR)))
        self.assertTrue(gate.update(packet(0.004)))

    def test_steady_background_is_not_speech_after_adaptation(self):
        gate = SpeechActivityGate()
        results = [gate.update(packet(0.003)) for _ in range(100)]
        self.assertFalse(results[-1])
        self.assertAlmostEqual(gate.floor, 0.003, places=3)

    def test_hysteresis_keeps_a_fading_tail_on(self):
        tail = packet(0.0022)
        gate = self.settled()
        self.assertFalse(gate.update(tail))
        for _ in range(5):
            gate.update(packet(0.01))
        self.assertTrue(gate.update(tail))
        self.assertFalse(gate.update(packet(QUIET_FLOOR)))

    def test_a_loud_room_raises_the_floor_until_noise_is_not_speech(self):
        gate = SpeechActivityGate()
        results = [gate.update(packet(0.02)) for _ in range(600)]
        self.assertFalse(results[-1])
        self.assertGreater(gate.floor, 0.015)
        self.assertTrue(gate.update(packet(0.2)))


if __name__ == "__main__":
    unittest.main()
