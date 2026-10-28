import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.config.live import (
    GEOMETRY_EXPIRY_SECONDS,
    IDENTITY_ABSENCE_SECONDS,
    SILENT_EVIDENCE_DECAY_SECONDS,
    UNMEASURED_CARRYOVER_SECONDS,
)
from src.config.models import MODEL_DIRECTORY
from src.core.contracts import FaceBox, Word
from src.core.tracking.realtime import RealtimeTracker, SpeakerEstimate
from src.core.vision.live import LiveFaceDetector, face_roll_degrees


class RealtimeTrackingTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.core.tracking.realtime.MINIMUM_FACE_TRACK_FRAMES", 1)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tracker = RealtimeTracker()
        self.speech = np.full(1600, 0.05, dtype=np.float32)

    TALK = [(0.1 * step, 0.09 if step % 2 else 0.02) for step in range(1, 13)]

    def faces(self, timestamp, aperture, second_aperture=0.0):
        return self.tracker.update_faces(
            [
                (FaceBox(timestamp, 20, 10, 40, 40), aperture),
                (FaceBox(timestamp, 100, 10, 40, 40), second_aperture),
            ],
            timestamp,
        )

    def history(self, *labels):
        """labels: (speaker, state) per 100 ms estimate starting at 0.0."""
        self.tracker.speaker_history.extend(
            SpeakerEstimate(i / 10, speaker, None, state, 1.0)
            for i, (speaker, state) in enumerate(labels)
        )

    def words(self, start, end, count=4):
        step = (end - start) / count
        return [Word(f"w{i}", start + i * step, start + (i + 1) * step, 1.0) for i in range(count)]

    def test_utterance_gets_one_speaker_despite_flicker(self):
        self.history(*[("Person 1", "visible"), ("Person 1", "visible"), ("Uncertain speaker", "uncertain"), ("Person 5", "visible")] * 4)
        events = self.tracker.caption_events(self.words(0.0, 1.1))
        self.assertEqual([e.speaker for e in events], ["Person 1"])
        self.assertEqual(events[0].text, "w0 w1 w2 w3")

    def test_named_person_outranks_generic_with_fair_share(self):
        self.history(*[("Person 2", "visible")] * 4, *[("Uncertain speaker", "uncertain")] * 6)
        self.assertEqual(self.tracker.caption_events(self.words(0.0, 0.9))[0].speaker, "Person 2")

    def test_clear_long_stretch_splits_utterance(self):
        self.history(*[("Person 1", "visible")] * 20, *[("Person 2", "visible")] * 20)
        events = self.tracker.caption_events(self.words(0.0, 3.9, 8))
        self.assertEqual([e.speaker for e in events], ["Person 1", "Person 2"])

    def test_weak_vote_reuses_recent_named_speaker(self):
        self.history(*[("Person 1", "visible")] * 10)
        self.tracker.caption_events(self.words(0.0, 0.9))
        self.tracker.speaker_history.extend(
            SpeakerEstimate(1.5 + i / 10, "Uncertain speaker", None, "uncertain", 0.0)
            for i in range(10)
        )
        self.assertEqual(self.tracker.caption_events(self.words(1.5, 2.4))[0].speaker, "Person 1")
        self.tracker.speaker_history.extend(
            SpeakerEstimate(9 + i / 10, "Uncertain speaker", None, "uncertain", 0.0)
            for i in range(10)
        )
        self.assertEqual(
            self.tracker.caption_events(self.words(9.0, 9.9))[0].speaker, "Uncertain speaker"
        )

    def voiced(self, face_a, face_b, utterances):
        """utterances: (start, end, voice, face voted) with visible 'face' estimates."""
        self.tracker.labels = {"face-1": 1, "face-2": 2}
        self.tracker.current_faces = [
            SimpleNamespace(track_id="face-1"), SimpleNamespace(track_id="face-2")
        ]
        results = []
        for start, end, voice, face in utterances:
            self.tracker.voice_timeline.append((start, end, voice))
            self.tracker.speaker_history.extend(
                SpeakerEstimate(start + i / 10, f"Person {face[-1]}", face, "visible", 1.0)
                for i in range(int((end - start) * 10))
            )
            results.append(self.tracker.caption_events(self.words(start, end))[0])
        return results

    def test_voice_with_enough_evidence_overrides_frame_votes(self):
        first, second = self.voiced("face-1", "face-2", [(0, 10, "v", "face-1"), (10, 12, "v", "face-2")])
        self.assertEqual(second.speaker, "Person 1")

    def test_voice_without_enough_evidence_keeps_frame_votes(self):
        first, second = self.voiced("face-1", "face-2", [(0, 5, "v", "face-1"), (5, 7, "v", "face-2")])
        self.assertEqual(second.speaker, "Person 2")

    def test_each_voice_follows_its_own_face(self):
        events = self.voiced("face-1", "face-2", [
            (0, 9, "a", "face-1"), (9, 18, "b", "face-2"),
            (18, 20, "a", "face-2"), (20, 22, "b", "face-1"),
        ])
        self.assertEqual([e.speaker for e in events[2:]], ["Person 1", "Person 2"])

    def test_one_person_split_over_two_voice_ids_still_overrides(self):
        self.voiced("face-1", "face-2", [(0, 9, "a", "face-1"), (9, 18, "b", "face-1")])
        self.tracker.voice_timeline.extend([(18, 20, "a"), (20, 22, "b")])
        self.tracker.speaker_history.extend(
            SpeakerEstimate(18 + i / 10, "Person 2", "face-2", "visible", 1.0)
            for i in range(40)
        )
        event = self.tracker.caption_events(self.words(18, 22))[0]
        self.assertEqual(event.speaker, "Person 1")

    def live(self, votes, aperture=0.09, second_aperture=0.0, leader_present=True):
        """Face 1 and 2 on screen, voice 'v' active; face 2's mouth moves when asked."""
        from collections import Counter
        for t, a, b in [(0.1, 0.02, 0.02), (0.2, aperture, second_aperture), (0.3, 0.03, 0.03)]:
            self.tracker.update_audio(self.speech, t)
            self.faces(t, a, b)
        self.tracker.voice_timeline.append((0.0, 0.9, "v"))
        self.tracker.voice_turn_votes["v"] = Counter(votes)
        # Speaking evidence integrates over seconds, so face 2 talks for 1.2 s.
        for step in range(4, 16):
            t = step / 10
            self.tracker.update_audio(self.speech, t)
            estimate = self.faces(t, 0.02, 0.10 if step % 2 else 0.03)
        return estimate

    def test_live_estimate_follows_voice_leader_over_mouth_motion(self):
        estimate = self.live({"face-1": 9.0})
        self.assertEqual((estimate.state, estimate.face_track), ("visible", "face-1"))
        self.assertTrue(estimate.voiced)
        self.assertNotIn("voiced", estimate.as_json())

    def test_live_estimate_without_leader_uses_mouth_motion(self):
        estimate = self.live({"face-1": 3.0})
        self.assertFalse(estimate.voiced)
        self.assertNotEqual(estimate.face_track, None)

    def test_off_screen_leader_is_named_off_screen(self):
        self.live({"face-1": 3.0})
        self.tracker.voice_turn_votes["v"]["face-9"] = 9.0
        self.tracker.labels["face-9"] = 9
        self.tracker.voice_timeline.append((4.0, 5.0, "v"))
        estimate = self.tracker.update_audio(self.speech, 5.0)
        self.assertEqual((estimate.speaker, estimate.state, estimate.face_track), ("Person 9", "off_screen", None))

    def test_voiced_estimates_tally_their_mouth_motion_face_not_the_override(self):
        self.live({"face-1": 9.0})
        history = [item for item in self.tracker.speaker_history if item.voiced]
        self.assertTrue(any(item.raw_face == "face-2" for item in history))
        self.assertNotIn("raw_face", history[-1].as_json())
        self.tracker.voice_timeline.append((0.0, 0.9, "v"))
        self.tracker.caption_events(self.words(0.4, 1.5))
        tally = self.tracker.voice_turn_votes["v"]
        self.assertGreater(tally["face-2"], 0.0)  # so a wrong leader can still be corrected

    def test_merge_remaps_voice_turn_votes(self):
        self.tracker.voice_turn_votes = {"v": {"face-2": 3.0, "face-1": 1.0}}
        self.tracker.voice_turn_votes["v"] = __import__("collections").Counter(self.tracker.voice_turn_votes["v"])
        self.tracker.boxes = {"face-1": FaceBox(0, 0, 0, 1, 1), "face-2": FaceBox(0, 0, 0, 1, 1)}
        self.tracker.mouths = {"face-1": [], "face-2": []}
        self.tracker.labels = {"face-1": 1}
        self.tracker._merge_identity("face-2", "face-1")
        self.assertEqual(dict(self.tracker.voice_turn_votes["v"]), {"face-1": 4.0})

    def test_audio_and_mouth_motion_identify_person_without_transcription(self):
        for timestamp, aperture in self.TALK:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.faces(timestamp, aperture)
        self.assertEqual(
            (estimate.state, estimate.speaker, estimate.face_track),
            ("visible", "Person 1", "face-1"),
        )
        self.assertEqual(self.tracker.speaker_at(timestamp).face_track, "face-1")

    def test_mouth_motion_without_audio_does_not_highlight_a_speaker(self):
        for timestamp, aperture in [(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)]:
            estimate = self.faces(timestamp, aperture)
        self.assertEqual(estimate.state, "silent")

    def test_voice_without_visible_mouth_motion_is_off_screen(self):
        self.tracker.update_audio(self.speech, 0.1)
        self.assertEqual(self.faces(0.1, 0.0).state, "off_screen")
        self.assertEqual(self.tracker.update_faces([], 0.2).state, "off_screen")

    def test_competing_mouths_are_uncertain(self):
        for timestamp, aperture in self.TALK:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.faces(timestamp, aperture, aperture)
        self.assertEqual(estimate.state, "uncertain")
        self.assertIsNone(estimate.face_track)

    def test_audio_silence_releases_highlight_and_stale_geometry_is_uncertain(self):
        self.tracker.update_audio(self.speech, 0.1)
        self.faces(0.1, 0.05)
        estimate = self.tracker.update_audio(np.zeros(1600), 0.4)
        self.assertEqual(estimate.state, "silent")
        estimate = self.tracker.update_audio(
            self.speech, 0.1 + GEOMETRY_EXPIRY_SECONDS + 0.01
        )
        self.assertEqual(estimate.state, "uncertain")

    def test_face_miss_retains_only_identity_and_expiry_creates_new_person(self):
        self.faces(0.1, 0.02)
        self.tracker.update_faces([], 0.2)
        self.assertEqual(self.tracker.current_faces, [])
        self.faces(0.3, 0.05)
        self.assertEqual(self.tracker.current_faces[0].track_id, "face-1")
        self.faces(0.3 + IDENTITY_ABSENCE_SECONDS + 0.01, 0.03)
        self.assertNotEqual(self.tracker.current_faces[0].track_id, "face-1")

    def test_missing_landmarks_do_not_create_artificial_mouth_motion(self):
        self.tracker.update_audio(self.speech, 0.1)
        self.faces(0.1, 0.08)
        self.faces(0.2, 0.08)
        self.tracker.update_faces([(FaceBox(0.3, 20, 10, 40, 40), None)], 0.3)
        self.assertEqual(self.tracker.current_faces[0].mouth_motion, 0.0)
        estimate = self.faces(0.4, 0.08)
        self.assertNotEqual(estimate.state, "visible")

    def test_out_of_order_frames_do_not_change_identity(self):
        self.faces(1.0, 0.02)
        with self.assertRaises(ValueError):
            self.faces(0.5, 0.09)
        self.assertEqual(self.tracker.latest_vision, 1.0)

    def subtle(self, timestamp, level, second_level=None, sizes=(40, 40)):
        observations = [(FaceBox(timestamp, 20, 10, sizes[0], sizes[0]), 0.05 + level)]
        if second_level is not None:
            observations.append((FaceBox(timestamp, 100, 10, sizes[1], sizes[1]), 0.05 + second_level))
        return self.tracker.update_faces(observations, timestamp)

    def test_single_subtly_moving_mouth_is_attributed(self):
        for step in range(1, 26):
            self.tracker.update_audio(self.speech, step / 10)
            estimate = self.subtle(step / 10, 0.02 * (step % 2))
        self.assertEqual((estimate.state, estimate.face_track), ("visible", "face-1"))
        self.assertGreater(estimate.confidence, 0.4)

    def test_two_similarly_small_movers_stay_uncertain(self):
        for step in range(1, 26):
            self.tracker.update_audio(self.speech, step / 10)
            estimate = self.subtle(step / 10, 0.02 * (step % 2), 0.02 * (step % 2))
        self.assertEqual(estimate.state, "uncertain")

    def test_jitter_baseline_learned_in_silence_discounts_motion(self):
        for step in range(1, 41):
            self.subtle(step / 10, 0.02 * (step % 2), 0.0)
        self.assertGreater(self.tracker.mouth_baselines["face-1"], 0.015)
        for step in range(41, 66):
            self.tracker.update_audio(self.speech, step / 10)
            estimate = self.subtle(step / 10, 0.025 * (step % 2), 0.03 * (step % 2))
        self.assertEqual((estimate.state, estimate.face_track), ("visible", "face-2"))

    def test_baseline_is_not_learned_while_speaking_and_is_dropped_with_the_face(self):
        for step in range(1, 6):
            self.tracker.update_audio(self.speech, step / 10)
            self.subtle(step / 10, 0.02 * (step % 2))
        self.assertEqual(self.tracker.mouth_baselines, {})
        for step in range(8, 12):
            self.subtle(step / 10, 0.0)
        self.assertIn("face-1", self.tracker.mouth_baselines)
        self.tracker.update_faces([], 1.2)
        self.assertEqual(self.tracker.mouth_baselines, {})

    def test_small_face_is_not_zeroed_by_size_weighting(self):
        for step in range(1, 26):
            self.tracker.update_audio(self.speech, step / 10)
            estimate = self.subtle(step / 10, 0.0, 0.04 * (step % 2), sizes=(40, 12))
        self.assertEqual((estimate.state, estimate.face_track), ("visible", "face-2"))

    def speak(self, steps, second_aperture=0.0):
        estimate = None
        for timestamp, aperture in steps:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.faces(timestamp, aperture, second_aperture)
        return estimate

    def test_unmeasurable_mouth_while_speaking_is_not_off_screen(self):
        self.speak(self.TALK)
        self.tracker.update_audio(self.speech, 1.6)
        estimate = self.tracker.update_faces(
            [(FaceBox(1.6, 20, 10, 40, 40), None)], 1.6
        )
        self.assertEqual((estimate.state, estimate.face_track), ("visible", "face-1"))
        fresh = RealtimeTracker()
        fresh.update_audio(self.speech, 0.1)
        estimate = fresh.update_faces([(FaceBox(0.1, 20, 10, 40, 40), None)], 0.1)
        self.assertEqual(estimate.state, "uncertain")

    def test_one_dropped_mesh_frame_does_not_split_speaker(self):
        apertures = [0.02, 0.09] * 6 + [None] + [0.02, 0.09] * 3
        states = []
        for index, aperture in enumerate(apertures):
            timestamp = round(0.1 * (index + 1), 1)
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.tracker.update_faces(
                [(FaceBox(timestamp, 20, 10, 40, 40), aperture)], timestamp
            )
            states.append((estimate.state, estimate.face_track))
        self.assertEqual(set(states[8:]), {("visible", "face-1")})

    def test_brief_challenger_does_not_take_over_but_sustained_one_does(self):
        self.speak(self.TALK)
        self.assertEqual(self.tracker.active_identity, "face-1")
        for step in range(13, 16):
            timestamp = step / 10
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.tracker.update_faces(
                [
                    (FaceBox(timestamp, 20, 10, 40, 40), 0.05),
                    (FaceBox(timestamp, 100, 10, 40, 40), 0.02 + 0.2 * (step % 2)),
                ],
                timestamp,
            )
        self.assertEqual(estimate.face_track, "face-1")
        for step in range(16, 60):
            timestamp = step / 10
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.tracker.update_faces(
                [
                    (FaceBox(timestamp, 20, 10, 40, 40), 0.05),
                    (FaceBox(timestamp, 100, 10, 40, 40), 0.02 + 0.2 * (step % 2)),
                ],
                timestamp,
            )
        self.assertEqual(estimate.face_track, "face-2")

    def test_measured_quiet_faces_release_to_off_screen_after_hold(self):
        self.speak(self.TALK)
        for step in range(13, 60):
            estimate = self.speak([(step / 10, 0.05)])
        self.assertEqual(estimate.state, "off_screen")

    def test_evidence_accumulates_during_speech_and_decays_in_silence(self):
        values = []
        for timestamp, aperture in self.TALK:
            self.tracker.update_audio(self.speech, timestamp)
            self.faces(timestamp, aperture)
            values.append(self.tracker.speaking_evidence["face-1"])
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[3])
        silence = np.zeros(1600, dtype=np.float32)
        decayed = {}
        for step in range(13, 33):
            self.tracker.update_audio(silence, step / 10)
            self.faces(step / 10, 0.05)
            decayed[step] = self.tracker.speaking_evidence["face-1"]
        expected = decayed[22] * np.exp(-1.0 / SILENT_EVIDENCE_DECAY_SECONDS)
        self.assertAlmostEqual(decayed[32], expected, places=6)
        self.assertLess(decayed[32], values[-1])
        self.assertEqual(self.tracker.speaking_evidence["face-2"], 0.0)

    def test_steady_speaker_beats_a_listener_with_one_short_burst(self):
        for step in range(1, 41):
            timestamp = step / 10
            burst = 0.08 * (step % 2) if step in (30, 31) else 0.0
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.subtle(timestamp, 0.04 * (step % 2), burst)
            if step == 33:
                self.assertGreater(
                    self.tracker.current_faces[1].mouth_motion,
                    self.tracker.current_faces[0].mouth_motion,
                )
        self.assertEqual((estimate.state, estimate.face_track), ("visible", "face-1"))

    def test_equal_steady_movers_stay_uncertain(self):
        for timestamp, aperture in self.TALK:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.faces(timestamp, aperture, aperture)
        self.assertAlmostEqual(
            self.tracker.speaking_evidence["face-1"],
            self.tracker.speaking_evidence["face-2"],
        )
        self.assertEqual(estimate.state, "uncertain")

    def test_clear_speaker_is_recorded_in_a_crowd_unless_a_rival_is_close(self):
        def run(rival_level):
            self.setUp()
            for step in range(1, 31):
                timestamp = step / 10
                self.tracker.update_audio(self.speech, timestamp)
                self.subtle(timestamp, 0.04 * (step % 2), rival_level * (step % 2))
            return {face for _, face in self.tracker.clear_speakers}

        self.assertEqual(run(0.015), {"face-1"})
        self.assertEqual(run(0.025), set())

    def test_oversized_image_is_rejected_before_opencv_allocation(self):
        content = BytesIO()
        Image.new("RGB", (641, 480)).save(content, format="JPEG")
        with (
            patch("src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"),
            patch(
                "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
            ),
            patch("src.core.vision.live.cv2.imdecode") as decode,
            patch("src.core.vision.live.face_identity_encoder"),
        ):
            detector = LiveFaceDetector()
            with self.assertRaises(ValueError):
                detector.detect(content.getvalue(), 0)
        decode.assert_not_called()


class UnmeasuredCarryOverTest(unittest.TestCase):
    def setUp(self):
        self.tracker = RealtimeTracker()
        self.speech = np.full(1600, 0.05, dtype=np.float32)

    def step(self, index, aperture):
        timestamp = round(0.1 * index, 1)
        self.tracker.update_audio(self.speech, timestamp)
        return self.tracker.update_faces(
            [(FaceBox(timestamp, 20, 10, 40, 40), aperture)], timestamp
        )

    def run_steps(self, first, last, aperture):
        return {index: self.step(index, aperture) for index in range(first, last + 1)}

    def test_carry_over_holds_then_expires_and_never_becomes_off_screen(self):
        for index in range(1, 13):
            self.step(index, 0.09 if index % 2 else 0.02)
        estimates = self.run_steps(13, 40, None)
        unmeasured_start = 15
        cap_steps = round(UNMEASURED_CARRYOVER_SECONDS * 10)
        for index in range(unmeasured_start, unmeasured_start + 10 + 1):
            self.assertEqual(estimates[index].state, "visible", index)
        for index in range(unmeasured_start + cap_steps + 1, 41):
            self.assertEqual(estimates[index].state, "uncertain", index)

    def test_measured_frames_reset_the_carry_over_clock(self):
        for index, aperture in [(1, 0.02), (2, 0.09), (3, 0.03)]:
            self.step(index, aperture)
        self.run_steps(4, 16, None)
        for index, aperture in [(17, 0.02), (18, 0.09), (19, 0.03)]:
            self.step(index, aperture)
        estimates = self.run_steps(20, 32, None)
        for index in range(22, 32):
            self.assertEqual(estimates[index].state, "visible", index)


class LiveFaceFeaturesTest(unittest.TestCase):
    def setUp(self):
        content = BytesIO()
        Image.new("RGB", (320, 240)).save(content, format="JPEG")
        self.content = content.getvalue()
        self.detection = SimpleNamespace(
            location_data=SimpleNamespace(
                relative_bounding_box=SimpleNamespace(
                    xmin=0.2, ymin=0.2, width=0.3, height=0.4
                )
            )
        )

    def test_tilted_face_is_rotated_upright_before_mesh_and_encoder(self):
        tilted = SimpleNamespace(
            location_data=SimpleNamespace(
                relative_bounding_box=self.detection.location_data.relative_bounding_box,
                relative_keypoints=[
                    SimpleNamespace(x=0.3, y=0.3),
                    SimpleNamespace(x=0.4, y=0.4),
                ],
            )
        )
        self.assertAlmostEqual(face_roll_degrees(tilted, 320, 240), 36.87, places=1)
        crops = []
        for detection in (self.detection, tilted):
            with (
                patch(
                    "src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"
                ) as mesh,
                patch(
                    "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
                ) as detector_class,
                patch("src.core.vision.live.face_identity_encoder"),
            ):
                detector_class.return_value.process.return_value.detections = [
                    detection
                ]
                mesh.return_value.process.return_value.multi_face_landmarks = []
                detector = LiveFaceDetector()
                self.addCleanup(detector.close)
                image = Image.effect_noise((320, 240), 60).convert("RGB")
                content = BytesIO()
                image.save(content, format="PNG")
                detector.detect(content.getvalue(), 0.1)
                crops.append(mesh.return_value.process.call_args.args[0])
        self.assertEqual(crops[0].shape, crops[1].shape)
        self.assertFalse(np.array_equal(crops[0], crops[1]))

    def test_current_frame_passes_crop_identity_features_to_tracker(self):
        features = np.ones(128, dtype=np.float32)
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
        with (
            patch(
                "src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"
            ) as mesh,
            patch(
                "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
            ) as detection,
            patch("src.core.vision.live.face_identity_encoder", create=True) as encoder,
        ):
            detection.return_value.process.return_value.detections = [self.detection]
            mesh.return_value.process.return_value.multi_face_landmarks = [
                SimpleNamespace(landmark=landmarks)
            ]
            encoder.return_value.encode.return_value = features
            detector = LiveFaceDetector()
            self.addCleanup(detector.close)
            width, height, observations = detector.detect(self.content, 0.1)
            self.assertEqual((width, height), (320, 240))
            self.assertIs(observations[0].features, features)
            encoder.assert_called_once_with(MODEL_DIRECTORY)
            crop, extracted_landmarks = encoder.return_value.encode.call_args.args
            self.assertGreater(crop.size, 0)
            self.assertIs(extracted_landmarks, landmarks)

    def test_missing_mesh_keeps_face_box_without_identity_or_mouth_features(self):
        with (
            patch(
                "src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"
            ) as mesh,
            patch(
                "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
            ) as detection,
            patch("src.core.vision.live.face_identity_encoder", create=True) as encoder,
        ):
            detection.return_value.process.return_value.detections = [self.detection]
            mesh.return_value.process.return_value.multi_face_landmarks = []
            detector = LiveFaceDetector()
            self.addCleanup(detector.close)
            _, _, observations = detector.detect(self.content, 0.1)
            self.assertIsNone(observations[0].features)
            self.assertIsNone(observations[0].aperture)
            encoder.return_value.encode.assert_not_called()
