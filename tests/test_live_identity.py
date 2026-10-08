import unittest
from unittest.mock import patch

import numpy as np

from src.config.live import IDENTITY_ABSENCE_SECONDS
from src.core.contracts import FaceBox
from src.core.tracking.realtime import RealtimeTracker


def observation(timestamp, position, features, aperture=0.02):
    return FaceBox(timestamp, position, 10, 40, 40), aperture, features


class LiveIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tracker = RealtimeTracker()
        self.first_features = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.second_features = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.return_time = IDENTITY_ABSENCE_SECONDS + 10.0

    def update(self, timestamp, *observations):
        self.tracker.update_faces(list(observations), timestamp)
        return [face.track_id for face in self.tracker.current_faces]

    def test_return_after_geometry_expiry_restores_person_at_another_position(self):
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        self.update(1.0)
        returned = self.update(
            self.return_time,
            observation(self.return_time, 300, self.first_features * 5),
        )
        self.assertEqual(returned, first)

    def test_different_person_at_same_position_does_not_inherit_identity(self):
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        self.update(0.2)
        replacement = self.update(0.3, observation(0.3, 20, self.second_features))
        self.assertNotEqual(replacement, first)
        returned = self.update(0.4, observation(0.4, 200, self.first_features))
        self.assertEqual(returned, first)

    def test_two_returning_people_keep_ids_when_positions_and_order_change(self):
        first = self.update(
            0.1,
            observation(0.1, 20, self.first_features),
            observation(0.1, 200, self.second_features),
        )
        self.update(1.0)
        returned = self.update(
            self.return_time,
            observation(self.return_time, 20, self.second_features),
            observation(self.return_time, 200, self.first_features),
        )
        self.assertEqual(returned, first[::-1])

    def test_visible_crossing_uses_features_instead_of_boxes(self):
        first = self.update(
            0.1,
            observation(0.1, 20, self.first_features),
            observation(0.1, 200, self.second_features),
        )
        crossed = self.update(
            0.2,
            observation(0.2, 20, self.second_features),
            observation(0.2, 200, self.first_features),
        )
        self.assertEqual(crossed, first[::-1])

    def test_identity_is_assigned_at_most_once_per_frame(self):
        self.update(0.1, observation(0.1, 20, self.first_features))
        current = self.update(
            0.2,
            observation(0.2, 20, self.first_features),
            observation(0.2, 200, self.first_features),
        )
        self.assertEqual(len(set(current)), 2)

    def test_ambiguous_reentry_does_not_guess_between_known_people(self):
        known = self.update(
            0.1,
            observation(0.1, 20, self.first_features),
            observation(0.1, 200, self.second_features),
        )
        ambiguous = self.update(
            self.return_time,
            observation(self.return_time, 20, np.array([1.0, 1.0, 0.0])),
        )
        self.assertNotIn(ambiguous[0], known)

    def test_off_screen_speech_and_fresh_mouth_history_on_return(self):
        speech = np.full(1600, 0.05, dtype=np.float32)
        for timestamp, aperture in [(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)]:
            self.tracker.update_audio(speech, timestamp)
            self.update(
                timestamp, observation(timestamp, 20, self.first_features, aperture)
            )
        self.assertEqual(self.tracker.estimate(0.3).speaker, "Person 1")
        self.tracker.update_audio(speech, 0.4)
        self.update(0.4)
        self.assertEqual(self.tracker.estimate(0.4).state, "off_screen")
        for offset, aperture in [(0.0, 0.03), (0.1, 0.09), (0.2, 0.02)]:
            timestamp = self.return_time + offset
            self.tracker.update_audio(speech, timestamp)
            self.update(
                timestamp, observation(timestamp, 300, self.first_features, aperture)
            )
            if offset == 0.0:
                self.assertEqual(self.tracker.current_faces[0].mouth_motion, 0.0)
        self.assertEqual(self.tracker.estimate(timestamp).speaker, "Person 1")

    def test_separate_sessions_do_not_share_face_features(self):
        self.update(0.1, observation(0.1, 20, self.first_features))
        self.update(0.2, observation(0.2, 20, self.second_features))
        fresh = RealtimeTracker()
        fresh.update_faces([observation(0.1, 20, self.second_features)], 0.1)
        self.assertEqual(fresh.current_faces[0].track_id, "face-1")

    def test_missing_features_preserve_short_continuity_but_cannot_reidentify(self):
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        continued = self.update(0.2, observation(0.2, 20, None))
        self.assertEqual(continued, first)
        returned = self.update(
            self.return_time, observation(self.return_time, 20, None)
        )
        self.assertNotEqual(returned, first)

    def test_invalid_features_are_ignored_and_do_not_poison_gallery(self):
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        for timestamp, features in [(0.2, np.zeros(3)), (0.3, np.full(3, np.nan))]:
            self.assertEqual(
                self.update(timestamp, observation(timestamp, 20, features)), first
            )
        returned = self.update(
            self.return_time, observation(self.return_time, 200, self.first_features)
        )
        self.assertEqual(returned, first)

    def test_faces_without_features_cannot_steal_a_feature_matched_identity(self):
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        current = self.update(
            0.2,
            observation(0.2, 20, None),
            observation(0.2, 200, self.first_features),
        )
        self.assertEqual(current[1], first[0])
        self.assertNotEqual(current[0], first[0])

    def test_bounded_gallery_evicts_oldest_dormant_identity(self):
        with patch("src.core.tracking.identity.MAXIMUM_SESSION_IDENTITIES", 2):
            first = self.update(0.1, observation(0.1, 20, self.first_features))
            second = self.update(0.2, observation(0.2, 200, self.second_features))
            self.update(0.3, observation(0.3, 300, np.array([0.0, 0.0, 1.0])))
            returned = self.update(
                self.return_time,
                observation(self.return_time, 20, self.second_features),
            )
            self.assertEqual(returned, second)
            forgotten = self.update(
                self.return_time + 1,
                observation(self.return_time + 1, 200, self.first_features),
            )
            self.assertNotEqual(forgotten, first)

    def test_feature_gallery_takes_a_copy_of_observed_features(self):
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        self.first_features[:] = self.second_features
        returned = self.update(
            self.return_time,
            observation(self.return_time, 300, np.array([1.0, 0.0, 0.0])),
        )
        self.assertEqual(returned, first)

    def test_invalid_frame_timestamp_does_not_update_feature_gallery(self):
        first = self.update(1.0, observation(1.0, 20, self.first_features))
        with self.assertRaises(ValueError):
            self.update(0.5, observation(0.5, 20, self.second_features))
        returned = self.update(
            self.return_time,
            observation(self.return_time, 300, self.first_features),
        )
        self.assertEqual(returned, first)

    def test_brief_absence_does_not_restore_old_speaking_evidence(self):
        speech = np.full(1600, 0.05, dtype=np.float32)
        for timestamp, aperture in [(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)]:
            self.tracker.update_audio(speech, timestamp)
            self.update(
                timestamp, observation(timestamp, 20, self.first_features, aperture)
            )
        self.assertEqual(self.tracker.estimate(0.3).speaker, "Person 1")
        self.update(0.4)
        self.tracker.update_audio(speech, 0.5)
        self.update(0.5, observation(0.5, 200, self.first_features, 0.03))
        self.assertEqual(self.tracker.current_faces[0].track_id, "face-1")
        self.assertEqual(self.tracker.current_faces[0].mouth_motion, 0.0)
        self.assertEqual(self.tracker.estimate(0.5).state, "off_screen")

    def test_weak_but_consistent_embedding_keeps_identity_at_same_position(self):
        weak = np.array([0.35, 0.94, 0.0], dtype=np.float32)
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        second = self.update(0.2, observation(0.2, 20, weak))
        self.assertEqual(second, first)
        self.assertEqual(len(self.tracker.identities.features[first[0]]), 2)

    def test_different_face_at_same_position_below_floor_is_new(self):
        other = np.array([0.1, 0.99, 0.0], dtype=np.float32)
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        second = self.update(0.2, observation(0.2, 20, other))
        self.assertNotEqual(second, first)

    def test_weak_match_is_not_claimed_when_two_faces_contest_the_box(self):
        weak = np.array([0.35, 0.94, 0.0], dtype=np.float32)
        first = self.update(0.1, observation(0.1, 20, self.first_features))
        second = self.update(
            0.2, observation(0.2, 20, weak), observation(0.2, 22, weak)
        )
        self.assertEqual(len(set(second)), 2)
        self.assertNotIn(first[0], second[1:])
