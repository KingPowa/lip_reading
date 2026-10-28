import unittest

import numpy as np

from src.config.live import SPEAKER_HOLD_SECONDS
from src.config.tracking import (
    VOICE_BIND_MIN_CONFIRMATIONS,
    VOICE_MERGE_MIN_CONFIRMATIONS,
    VOICE_STICKY_MAXIMUM_SECONDS,
    VOICE_VALIDITY_SECONDS,
)
from src.core.contracts import FaceBox, Word
from src.core.tracking.realtime import RealtimeTracker

PEOPLE = np.eye(4, dtype=np.float32)
VOICES = np.eye(3, dtype=np.float32)


class VoiceTrackingTest(unittest.TestCase):
    def setUp(self):
        self.tracker = RealtimeTracker()
        self.speech = np.full(1600, 0.05, dtype=np.float32)
        self.step = 0

    @property
    def now(self):
        return round(0.1 * self.step, 1)

    def tick(self, *faces, speaking=True, audio=True):
        self.step += 1
        if audio:
            samples = self.speech if speaking else np.zeros(1600, dtype=np.float32)
            self.tracker.update_audio(samples, self.now)
        observations = [
            (FaceBox(self.now, position, 10, 40, 40), aperture, PEOPLE[person])
            for person, position, aperture in faces
        ]
        return self.tracker.update_faces(observations, self.now)

    def talk(self):
        return 0.09 if self.step % 2 else 0.02

    def pair(self, speaker):
        quiet = 0.02
        first = self.talk() if speaker in ("a", "both") else quiet
        second = self.talk() if speaker in ("b", "both") else quiet
        return (0, 20, first), (1, 200, second)

    def voice_window(self, voice):
        return self.tracker.update_voice(VOICES[voice], self.now - 0.9, self.now)

    def teach(self, speaker, voice, windows=VOICE_BIND_MIN_CONFIRMATIONS):
        for _ in range(windows):
            for _ in range(10):
                self.tick(*self.pair(speaker))
            self.voice_window(voice)

    def settle(self, seconds=15.0):
        """Silent, still faces so earlier speaking evidence decays away."""
        for _ in range(round(seconds * 10)):
            self.tick(*self.pair("none"), speaking=False)

    def teach_both(self):
        self.teach("a", 0)
        self.settle()
        self.teach("b", 1)
        self.settle()

    def test_voice_binds_only_after_enough_clear_single_speaker_windows(self):
        self.teach("a", 0, windows=VOICE_BIND_MIN_CONFIRMATIONS - 1)
        self.assertIsNone(self.tracker.voices.face_for("voice-1"))
        self.teach("a", 0, windows=1)
        self.assertEqual(self.tracker.voices.face_for("voice-1"), "face-1")

    def test_soft_speaker_with_subtle_mouth_motion_binds_a_voice(self):
        for _ in range(VOICE_BIND_MIN_CONFIRMATIONS + 2):
            for _ in range(10):
                soft = 0.04 if self.step % 2 else 0.02
                estimate = self.tick((0, 20, soft), (1, 200, 0.02))
            self.voice_window(0)
        self.assertEqual(self.tracker.voices.face_for("voice-1"), "face-1")

    def test_voice_is_not_bound_from_competing_or_carried_over_estimates(self):
        for _ in range(VOICE_BIND_MIN_CONFIRMATIONS):
            for _ in range(10):
                self.tick(*self.pair("both"))
            self.voice_window(0)
        self.assertIsNone(self.tracker.voices.face_for("voice-1"))
        self.teach("a", 1, windows=2)
        for _ in range(VOICE_BIND_MIN_CONFIRMATIONS):
            for _ in range(10):
                self.tick((0, 20, None), (1, 200, 0.02))
            self.voice_window(1)
        self.assertIsNone(self.tracker.voices.face_for("voice-2"))

    def test_current_voice_breaks_a_tie_between_equally_moving_mouths(self):
        self.teach_both()
        for _ in range(round(VOICE_STICKY_MAXIMUM_SECONDS * 10) + 8):
            estimate = self.tick(*self.pair("both"))
        self.assertEqual(estimate.state, "uncertain")
        self.voice_window(1)
        for _ in range(5):
            estimate = self.tick(*self.pair("both"))
        self.assertEqual((estimate.state, estimate.speaker), ("visible", "Person 2"))
        for _ in range(round(VOICE_STICKY_MAXIMUM_SECONDS * 10) + 8):
            estimate = self.tick(*self.pair("both"))
        self.assertEqual(estimate.state, "uncertain")

    def test_voice_alone_cannot_make_a_silent_mouth_the_speaker(self):
        self.teach_both()
        for _ in range(round(SPEAKER_HOLD_SECONDS * 10) + 6):
            estimate = self.tick((0, 20, 0.02), (1, 200, 0.02))
        self.voice_window(0)
        estimate = self.tick((0, 20, 0.02), (1, 200, 0.02))
        self.assertNotEqual(estimate.state, "visible")

    def test_bound_voice_of_a_turned_away_person_keeps_their_label(self):
        self.teach_both()
        for _ in range(10):
            self.tick((1, 200, 0.02))
        self.voice_window(0)
        estimate = self.tick((1, 200, 0.02))
        self.assertEqual(
            (estimate.speaker, estimate.state, estimate.face_track),
            ("Person 1", "off_screen", None),
        )

    def leave(self, ticks=15):
        """Person 1 walks off; Person 2 stays visible but silent."""
        for _ in range(ticks):
            estimate = self.tick((1, 200, 0.02))
        return estimate

    def ambiguous_window(self, voice=0):
        vector = np.zeros(3, dtype=np.float32)
        vector[voice], vector[2] = 0.4, 0.9
        return self.tracker.update_voice(vector, self.now - 0.9, self.now)

    def test_ambiguous_windows_keep_an_absent_persons_label(self):
        self.teach_both()
        self.leave()
        self.assertEqual(self.ambiguous_window(), "voice-1")
        self.assertEqual(len(self.tracker.voices.centroids), 2)
        estimate = self.tick((1, 200, 0.02))
        self.assertEqual(
            (estimate.speaker, estimate.state, estimate.face_track),
            ("Person 1", "off_screen", None),
        )

    def test_label_persists_across_a_gap_in_continuous_speech_then_expires(self):
        self.teach_both()
        self.leave()
        self.ambiguous_window()
        for _ in range(round(VOICE_VALIDITY_SECONDS * 10) + 8):
            estimate = self.tick((1, 200, 0.02))
        self.assertEqual(estimate.speaker, "Person 1")
        for _ in range(4):
            self.tick((1, 200, 0.02), speaking=False)
        for _ in range(round(VOICE_VALIDITY_SECONDS * 10) + 2):
            estimate = self.tick((1, 200, 0.02))
        self.assertEqual((estimate.speaker, estimate.state), ("Off-screen speaker", "off_screen"))

    def test_sticky_label_ends_after_the_maximum_even_in_continuous_speech(self):
        self.teach_both()
        self.leave()
        self.ambiguous_window()
        for _ in range(round(VOICE_STICKY_MAXIMUM_SECONDS * 10) + 3):
            estimate = self.tick((1, 200, 0.02))
        self.assertEqual(estimate.speaker, "Off-screen speaker")

    def test_unmeasured_visible_face_does_not_block_the_absent_label(self):
        self.teach_both()
        self.leave()
        self.ambiguous_window()
        estimate = self.tick((1, 200, None))
        self.assertEqual(
            (estimate.speaker, estimate.state, estimate.face_track),
            ("Person 1", "off_screen", None),
        )

    def test_voice_match_relabels_earlier_generic_off_screen_captions(self):
        self.teach_both()
        self.leave()
        self.assertEqual(self.tracker.speaker_history[-1].speaker, "Off-screen speaker")
        before = self.tracker.revision
        self.voice_window(0)
        events = self.tracker.caption_events([Word("hi", self.now - 0.3, self.now - 0.1, 1.0)])
        self.assertEqual((events[0].speaker, events[0].state), ("Person 1", "off_screen"))
        self.assertGreater(self.tracker.revision, before)
        self.assertIsNone(events[0].face_track)

    def test_relabel_leaves_visible_and_uncertain_entries_alone(self):
        self.teach_both()
        for _ in range(10):
            self.tick(*self.pair("both"))
        self.leave(45)
        kept = [
            item for item in self.tracker.speaker_history
            if item.state in ("visible", "uncertain")
        ]
        generic = [
            item for item in self.tracker.speaker_history
            if item.speaker == "Off-screen speaker"
        ]
        self.assertTrue(kept and generic)
        self.voice_window(0)
        history = list(self.tracker.speaker_history)
        self.assertTrue(all(item in history for item in kept))
        inside = [item for item in generic if item.timestamp >= self.now - 0.9]
        outside = [item for item in generic if item.timestamp < self.now - 0.9]
        self.assertTrue(inside and outside)
        self.assertTrue(all(item not in history for item in inside))
        self.assertTrue(all(item in history for item in outside))

    def test_unknown_or_unbound_voice_stays_off_screen(self):
        self.teach_both()
        for _ in range(10):
            self.tick((1, 200, 0.02))
        self.tracker.update_voice(VOICES[2], self.now - 0.9, self.now)
        estimate = self.tick((1, 200, 0.02))
        self.assertEqual((estimate.speaker, estimate.state), ("Off-screen speaker", "off_screen"))

    def test_bound_voice_labels_a_visible_face_whose_mouth_is_unmeasurable(self):
        self.teach_both()
        for _ in range(12):
            estimate = self.tick((0, 20, None), (1, 200, 0.02))
        self.assertEqual(estimate.state, "uncertain")
        self.voice_window(0)
        estimate = self.tick((0, 20, None), (1, 200, 0.02))
        self.assertEqual(
            (estimate.speaker, estimate.state, estimate.face_track),
            ("Person 1", "visible", "face-1"),
        )
        self.assertLess(estimate.confidence, 1.0)

    def test_new_face_speaking_with_an_absent_persons_voice_inherits_their_identity(self):
        self.teach_both()
        for _ in range(10):
            self.tick((1, 200, 0.02))
        for _ in range(4):
            self.tick((1, 200, 0.02), (2, 400, self.talk()))
        labels = dict(self.tracker.labels)
        self.assertEqual(len(labels), 3)
        for _ in range(VOICE_MERGE_MIN_CONFIRMATIONS):
            self.tick((1, 200, 0.02), (2, 400, self.talk()))
            self.voice_window(0)
        estimate = self.tick((1, 200, 0.02), (2, 400, self.talk()))
        self.assertEqual(
            sorted(face.track_id for face in self.tracker.current_faces),
            ["face-1", "face-2"],
        )
        self.assertEqual(self.tracker.labels["face-1"], labels["face-1"])
        self.assertNotEqual(estimate.speaker, "Person 3")

    def test_merge_carries_speaking_evidence_to_the_inherited_identity(self):
        self.teach_both()
        for _ in range(10):
            self.tick((1, 200, 0.02))
        for _ in range(4):
            self.tick((1, 200, 0.02), (2, 400, self.talk()))
        self.assertGreater(self.tracker.speaking_evidence["face-3"], 0.0)
        pending = self.tracker.speaking_evidence["face-3"]
        self.assertNotIn("face-1", self.tracker.speaking_evidence)
        self.tracker.update_voice(VOICES[0], self.now - 0.9, self.now)
        self.tick((1, 200, 0.02), (2, 400, self.talk()))
        self.tracker.update_voice(VOICES[0], self.now - 0.9, self.now)
        self.assertEqual(self.tracker.merge_count, 1)
        self.assertNotIn("face-3", self.tracker.speaking_evidence)
        self.assertNotIn("face-3", self.tracker.evidence_times)
        self.assertGreater(self.tracker.speaking_evidence["face-1"], pending)

    def test_merge_needs_consecutive_agreeing_segments(self):
        self.teach_both()
        for _ in range(10):
            self.tick((1, 200, 0.02))
        for _ in range(4):
            self.tick((1, 200, 0.02), (2, 400, self.talk()))
        self.voice_window(0)
        self.tick((1, 200, 0.02), (2, 400, self.talk()))
        self.voice_window(1)
        self.tick((1, 200, 0.02), (2, 400, self.talk()))
        self.voice_window(0)
        self.assertIn("face-3", [face.track_id for face in self.tracker.current_faces])

    def test_competing_new_faces_are_never_merged(self):
        self.teach_both()
        for _ in range(10):
            self.tick((1, 200, 0.02))
        for _ in range(4):
            self.tick((1, 200, 0.02), (2, 400, self.talk()), (3, 500, self.talk()))
        for _ in range(VOICE_MERGE_MIN_CONFIRMATIONS + 1):
            self.tick((1, 200, 0.02), (2, 400, self.talk()), (3, 500, self.talk()))
            self.voice_window(0)
        ids = [face.track_id for face in self.tracker.current_faces]
        self.assertNotIn("face-1", ids)
        self.assertEqual(len(ids), 3)

    def test_visible_identity_is_never_merged_into(self):
        self.teach_both()
        for _ in range(4):
            self.tick((0, 20, 0.02), (1, 200, 0.02), (2, 400, self.talk()))
        for _ in range(VOICE_MERGE_MIN_CONFIRMATIONS + 1):
            self.tick((0, 20, 0.02), (1, 200, 0.02), (2, 400, self.talk()))
            self.voice_window(0)
        ids = sorted(face.track_id for face in self.tracker.current_faces)
        self.assertEqual(ids, ["face-1", "face-2", "face-3"])


if __name__ == "__main__":
    unittest.main()
