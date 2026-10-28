import unittest

import numpy as np

from src.config.live import (
    LIVE_VOICE_WINDOW_SECONDS,
    MAXIMUM_VOICE_BUFFER_SECONDS,
    VOICE_GAP_RESET_SECONDS,
)
from src.config.tracking import MAXIMUM_VOICES, VOICE_BIND_MIN_CONFIRMATIONS
from src.core.tracking.voice import SpeechBuffer, VoiceGallery


def unit(*values):
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class VoiceGalleryTest(unittest.TestCase):
    def setUp(self):
        self.gallery = VoiceGallery()

    def test_similar_embeddings_share_a_voice_and_distinct_ones_create_new(self):
        first = self.gallery.assign(unit(1, 0, 0))
        self.assertEqual(first, "voice-1")
        self.assertEqual(self.gallery.assign(unit(1, 0.1, 0)), first)
        self.assertEqual(self.gallery.assign(unit(0, 1, 0)), "voice-2")

    def test_embedding_close_to_two_voices_is_not_guessed(self):
        self.gallery.assign(unit(1, 0.9, 0))
        self.gallery.assign(unit(1, -0.9, 0))
        self.assertEqual(len(self.gallery.centroids), 2)
        self.assertIsNone(self.gallery.assign(unit(1, 0, 0)))
        self.assertEqual(len(self.gallery.centroids), 2)

    def test_poor_match_between_the_thresholds_neither_joins_nor_creates_a_voice(self):
        self.gallery.assign(unit(1, 0, 0))
        self.assertIsNone(self.gallery.assign(unit(0.35, 0.94, 0)))
        self.assertEqual(list(self.gallery.centroids), ["voice-1"])

    def test_voice_count_is_bounded(self):
        for index in range(MAXIMUM_VOICES):
            self.assertIsNotNone(self.gallery.assign(np.eye(MAXIMUM_VOICES)[index]))
        self.assertIsNone(self.gallery.assign(unit(*[1.0] * MAXIMUM_VOICES)))
        self.assertEqual(len(self.gallery.centroids), MAXIMUM_VOICES)

    def test_invalid_embeddings_are_ignored(self):
        self.assertIsNone(self.gallery.assign(np.array([np.nan, 1.0])))
        self.assertIsNone(self.gallery.assign(np.zeros(3)))

    def test_binding_needs_confirmations_and_a_dominant_face(self):
        voice = self.gallery.assign(unit(1, 0, 0))
        for _ in range(VOICE_BIND_MIN_CONFIRMATIONS - 1):
            self.gallery.bind(voice, "face-1")
        self.assertIsNone(self.gallery.face_for(voice))
        self.gallery.bind(voice, "face-1")
        self.assertEqual(self.gallery.face_for(voice), "face-1")
        self.assertEqual(self.gallery.voice_for("face-1"), voice)
        for _ in range(VOICE_BIND_MIN_CONFIRMATIONS):
            self.gallery.bind(voice, "face-2")
        self.assertIsNone(self.gallery.face_for(voice))
        self.assertIsNone(self.gallery.voice_for("face-1"))


class VoiceIdentifyTest(unittest.TestCase):
    def setUp(self):
        self.gallery = VoiceGallery()
        self.voice = self.gallery.assign(unit(1, 0, 0))
        self.other = self.gallery.assign(unit(0, 1, 0))

    def bind(self, voice, face):
        for _ in range(VOICE_BIND_MIN_CONFIRMATIONS):
            self.gallery.bind(voice, face)

    def snapshot(self):
        return (
            {key: value.copy() for key, value in self.gallery.centroids.items()},
            dict(self.gallery.counts),
            self.gallery.next_voice,
        )

    def test_ambiguous_embedding_identifies_a_bound_voice_without_mutation(self):
        self.bind(self.voice, "face-1")
        ambiguous = unit(0.4, 0, 0.9)
        self.assertIsNone(self.gallery.assign(ambiguous))
        before = self.snapshot()
        self.assertEqual(self.gallery.identify(ambiguous), self.voice)
        after = self.snapshot()
        self.assertEqual(before[1:], after[1:])
        for key, value in before[0].items():
            np.testing.assert_array_equal(value, after[0][key])

    def test_unbound_weak_or_tied_voices_are_not_identified(self):
        ambiguous = unit(0.4, 0, 0.9)
        self.assertIsNone(self.gallery.identify(ambiguous))
        self.bind(self.voice, "face-1")
        self.assertIsNone(self.gallery.identify(unit(0.2, 0, 0.98)))
        self.assertIsNone(self.gallery.identify(unit(0.4, 0.4, 0.8)))
        self.assertIsNone(self.gallery.identify(np.zeros(3)))
        self.assertEqual(len(self.gallery.centroids), 2)


class SpeechBufferTest(unittest.TestCase):
    def packet(self, level):
        return np.full(1600, level, dtype=np.float32)

    def fill(self, buffer, start_step, steps, level=0.05):
        for step in range(start_step, start_step + steps):
            buffer.add(self.packet(level), round(0.1 * (step + 1), 1), level > 0)

    def test_window_is_ready_after_enough_speech_and_returns_its_time_span(self):
        buffer = SpeechBuffer()
        self.fill(buffer, 0, 9)
        self.assertFalse(buffer.ready())
        self.fill(buffer, 9, 1)
        self.assertTrue(buffer.ready())
        samples, start, end = buffer.take()
        self.assertEqual(len(samples), round(LIVE_VOICE_WINDOW_SECONDS * 16000))
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(end, 1.0)
        self.assertFalse(buffer.ready())

    def test_silence_is_not_buffered_and_a_long_gap_starts_a_new_window(self):
        buffer = SpeechBuffer()
        self.fill(buffer, 0, 5)
        self.fill(buffer, 5, 10, level=0.0)
        self.fill(buffer, 15, 5)
        self.assertFalse(buffer.ready())
        self.assertLess(VOICE_GAP_RESET_SECONDS, 1.0)

    def test_backlog_is_capped_by_dropping_the_oldest_speech(self):
        buffer = SpeechBuffer()
        self.fill(buffer, 0, 50)
        samples, start, end = buffer.take()
        self.assertLessEqual(len(samples) / 16000, MAXIMUM_VOICE_BUFFER_SECONDS + 1e-6)
        self.assertAlmostEqual(end, 5.0)
        self.assertGreater(start, 1.5)
