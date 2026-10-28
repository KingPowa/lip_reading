import unittest

from src.core.contracts import FaceActivity, FaceBox, FaceTrack, SpeakerTurn, Word
from src.core.tracking.association import associate_speakers


class AssociationTest(unittest.TestCase):
    def recording_events(
        self, second_scores, second_voice="voice", speaker_visible=True
    ):
        turns = [
            SpeakerTurn("voice", 0.0, 9.0, 0.9),
            SpeakerTurn(second_voice, 10.0, 15.0, 0.9),
        ]
        activities = [FaceActivity("speaker", 0.0, 9.0, 0.9)]
        activities.extend(
            FaceActivity(track, 10.0, 15.0, value)
            for track, value in second_scores.items()
        )
        tracks = [
            FaceTrack(
                track,
                0.0,
                24.0,
                [
                    FaceBox(index / 5, 0, 0, 10, 10)
                    for index in range(121)
                    if track != "speaker" or speaker_visible or index / 5 <= 9.0
                ],
            )
            for track in ("speaker", "listener")
        ]
        return associate_speakers(
            turns, [], tracks, activities, use_recording_voice_evidence=True
        )

    def test_known_voice_resolves_brief_ambiguous_face_scores(self):
        events = self.recording_events({"speaker": 0.45, "listener": 0.5})
        self.assertEqual(events[1].speaker, "Person 1")
        self.assertEqual(events[1].face_track, "speaker")

    def test_known_voice_does_not_move_to_a_strong_competing_visible_face(self):
        events = self.recording_events({"speaker": 0.3, "listener": 0.9})
        self.assertEqual(events[1].state, "visible")
        self.assertEqual(events[1].face_track, "speaker")

    def test_new_voice_can_switch_to_another_speaker(self):
        events = self.recording_events(
            {"speaker": 0.3, "listener": 0.9}, second_voice="other"
        )
        self.assertEqual(events[1].speaker, "Person 2")
        self.assertEqual(events[1].face_track, "listener")

    def test_no_visual_evidence_does_not_keep_a_visible_highlight(self):
        events = self.recording_events({}, speaker_visible=False)
        self.assertEqual(events[1].speaker, "Person 1")
        self.assertEqual(events[1].state, "off_screen")
        self.assertIsNone(events[1].face_track)

    def test_future_voice_evidence_resolves_an_initial_ambiguous_face_vote(self):
        tracks = [
            FaceTrack(
                track,
                0.0,
                20.0,
                [FaceBox(index / 5, 0, 0, 10, 10) for index in range(101)],
            )
            for track in ("speaker", "listener")
        ]
        turns = [
            SpeakerTurn("voice", 0.0, 2.0, 0.9),
            SpeakerTurn("voice", 3.0, 18.0, 0.9),
        ]
        activities = [
            FaceActivity("speaker", 0.0, 2.0, 0.6),
            FaceActivity("listener", 0.0, 2.0, 0.65),
            FaceActivity("speaker", 3.0, 18.0, 0.9),
        ]
        events = associate_speakers(
            turns, [], tracks, activities, use_recording_voice_evidence=True
        )
        self.assertEqual([event.face_track for event in events], ["speaker", "speaker"])
        self.assertEqual([event.speaker for event in events], ["Person 1", "Person 1"])

    def test_grounded_visible_face_keeps_its_box_when_activity_is_weak(self):
        events = self.recording_events({"speaker": 0.1, "listener": 0.4})
        self.assertEqual(events[1].speaker, "Person 1")
        self.assertEqual(events[1].state, "visible")
        self.assertEqual(events[1].face_track, "speaker")

    def test_brief_detection_dropout_keeps_the_grounded_face_rectangle(self):
        turns = [
            SpeakerTurn("voice", 0.0, 9.0, 0.9),
            SpeakerTurn("voice", 10.0, 15.0, 0.9),
        ]
        boxes = [
            FaceBox(index / 5, 0, 0, 10, 10)
            for index in range(76)
            if not 10.0 < index / 5 < 11.4
        ]
        activities = [
            FaceActivity("speaker", 0.0, 9.0, 0.9),
            FaceActivity("speaker", 10.0, 15.0, 0.0),
        ]
        events = associate_speakers(
            turns,
            [],
            [FaceTrack("speaker", 0.0, 15.0, boxes)],
            activities,
            use_recording_voice_evidence=True,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1].face_track, "speaker")
        self.assertEqual((events[1].start, events[1].end), (10.0, 15.0))

    def test_long_detection_gap_has_no_invented_visible_face(self):
        turns = [
            SpeakerTurn("voice", 0.0, 9.0, 0.9),
            SpeakerTurn("voice", 10.0, 15.0, 0.9),
        ]
        boxes = [
            FaceBox(index / 5, 0, 0, 10, 10)
            for index in range(76)
            if index / 5 <= 9.0 or index / 5 >= 12.0
        ]
        activities = [FaceActivity("speaker", 0.0, 9.0, 0.9)]
        events = associate_speakers(
            turns,
            [],
            [FaceTrack("speaker", 0.0, 15.0, boxes)],
            activities,
            use_recording_voice_evidence=True,
        )
        self.assertEqual(
            [(event.start, event.end, event.state) for event in events],
            [
                (0.0, 9.0, "visible"),
                (10.0, 12.0, "off_screen"),
                (12.0, 15.0, "visible"),
            ],
        )
        self.assertIsNone(events[1].face_track)
        self.assertTrue(all(event.speaker == "Person 1" for event in events))

    def test_single_visible_face_is_labeled(self) -> None:
        tracks = [FaceTrack("face-1", 0.0, 2.0, [FaceBox(0.0, 0, 0, 10, 10)])]
        turns = [SpeakerTurn("SPEAKER_00", 0.2, 1.0, 0.9)]
        words = [Word("hello", 0.3, 0.6, 0.9)]
        activities = [FaceActivity("face-1", 0.2, 1.0, 0.75)]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual(events[0].speaker, "Person 1")
        self.assertEqual(events[0].state, "visible")

    def test_close_candidates_stay_uncertain(self) -> None:
        tracks = [
            FaceTrack("face-1", 0.0, 2.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("face-2", 0.0, 2.0, [FaceBox(0.0, 20, 0, 10, 10)]),
        ]
        turns = [SpeakerTurn("SPEAKER_00", 0.2, 1.0, 0.9)]
        words = [Word("hello", 0.3, 0.6, 0.9)]
        activities = [
            FaceActivity("face-1", 0.2, 1.0, 0.71),
            FaceActivity("face-2", 0.2, 1.0, 0.70),
        ]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual(events[0].state, "uncertain")

    def test_voice_cluster_retains_label_after_face_track_changes(self) -> None:
        tracks = [
            FaceTrack("face-1", 0.0, 1.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("face-2", 1.0, 2.0, [FaceBox(1.0, 0, 0, 10, 10)]),
        ]
        turns = [
            SpeakerTurn("SPEAKER_00", 0.2, 0.8, 0.9),
            SpeakerTurn("SPEAKER_00", 1.2, 1.8, 0.9),
        ]
        words = [Word("hello", 0.3, 0.6, 0.9), Word("again", 1.3, 1.6, 0.9)]
        activities = [
            FaceActivity("face-1", 0.2, 0.8, 0.75),
            FaceActivity("face-2", 1.2, 1.8, 0.75),
        ]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual([event.speaker for event in events], ["Person 1", "Person 1"])

    def test_voice_cluster_fragments_reuse_the_label_of_the_same_face(self) -> None:
        tracks = [FaceTrack("face-1", 0.0, 2.0, [FaceBox(0.0, 0, 0, 10, 10)])]
        turns = [
            SpeakerTurn("SPEAKER_00", 0.2, 0.8, 0.9),
            SpeakerTurn("SPEAKER_01", 1.2, 1.8, 0.9),
        ]
        words = [Word("hello", 0.3, 0.6, 0.9), Word("again", 1.3, 1.6, 0.9)]
        activities = [
            FaceActivity("face-1", 0.2, 0.8, 0.75),
            FaceActivity("face-1", 1.2, 1.8, 0.75),
        ]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual([event.speaker for event in events], ["Person 1", "Person 1"])

    def test_voice_label_stays_stable_across_live_windows(self) -> None:
        labels: dict[str, str] = {}
        first = associate_speakers(
            [SpeakerTurn("SPEAKER_00", 0.0, 1.0, 0.9)],
            [Word("hello", 0.1, 0.8, 0.9)],
            [FaceTrack("face-1", 0.0, 1.0, [FaceBox(0.0, 0, 0, 10, 10)])],
            [FaceActivity("face-1", 0.0, 1.0, 0.9)],
            labels,
        )
        second = associate_speakers(
            [SpeakerTurn("SPEAKER_00", 0.0, 1.0, 0.9)],
            [Word("again", 0.1, 0.8, 0.9)],
            [FaceTrack("face-2", 0.0, 1.0, [FaceBox(0.0, 20, 0, 10, 10)])],
            [FaceActivity("face-2", 0.0, 1.0, 0.9)],
            labels,
        )
        self.assertEqual(
            [first[0].speaker, second[0].speaker], ["Person 1", "Person 1"]
        )

    def test_off_screen_speech_keeps_its_text_without_a_highlight(self) -> None:
        events = associate_speakers(
            [SpeakerTurn("SPEAKER_00", 0.0, 1.0, 0.9)],
            [Word("hello", 0.1, 0.8, 0.9)],
            [],
            [],
        )
        self.assertEqual(events[0].state, "off_screen")
        self.assertEqual(events[0].text, "hello")

    def test_overlapping_visible_turns_remain_separate_active_speakers(self) -> None:
        tracks = [
            FaceTrack("face-1", 0.0, 2.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("face-2", 0.0, 2.0, [FaceBox(0.0, 20, 0, 10, 10)]),
        ]
        turns = [
            SpeakerTurn("SPEAKER_00", 0.2, 1.2, 0.9),
            SpeakerTurn("SPEAKER_01", 0.8, 1.8, 0.9),
        ]
        words = [Word("first", 0.3, 0.6, 0.9), Word("second", 1.0, 1.3, 0.9)]
        activities = [
            FaceActivity("face-1", 0.2, 1.2, 0.75),
            FaceActivity("face-2", 0.8, 1.8, 0.75),
        ]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual([event.face_track for event in events], ["face-1", "face-2"])
        self.assertEqual(
            [event.state for event in events], ["overlapping", "overlapping"]
        )

    def test_face_track_cannot_receive_two_person_labels(self) -> None:
        tracks = [
            FaceTrack("face-1", 0.0, 3.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("face-2", 0.0, 3.0, [FaceBox(0.0, 20, 0, 10, 10)]),
        ]
        turns = [
            SpeakerTurn("SPEAKER_00", 0.2, 0.8, 0.9),
            SpeakerTurn("SPEAKER_01", 1.2, 1.8, 0.9),
            SpeakerTurn("SPEAKER_01", 2.2, 2.8, 0.9),
        ]
        words = [
            Word("first", 0.3, 0.6, 0.9),
            Word("second", 1.3, 1.6, 0.9),
            Word("conflict", 2.3, 2.6, 0.9),
        ]
        activities = [
            FaceActivity("face-1", 0.2, 0.8, 0.75),
            FaceActivity("face-2", 1.2, 1.8, 0.75),
            FaceActivity("face-1", 2.2, 2.8, 0.75),
        ]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual(
            [event.speaker for event in events],
            ["Person 1", "Person 2", "Uncertain speaker"],
        )

    def test_same_person_cannot_have_two_active_boxes(self) -> None:
        tracks = [
            FaceTrack("face-1", 0.0, 2.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("face-2", 0.0, 2.0, [FaceBox(0.0, 20, 0, 10, 10)]),
        ]
        turns = [
            SpeakerTurn("SPEAKER_00", 0.2, 1.2, 0.9),
            SpeakerTurn("SPEAKER_00", 0.8, 1.8, 0.8),
        ]
        words = [Word("first", 0.3, 0.6, 0.9), Word("second", 1.0, 1.3, 0.9)]
        activities = [
            FaceActivity("face-1", 0.2, 1.2, 0.9),
            FaceActivity("face-2", 0.8, 1.8, 0.75),
        ]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual([event.state for event in events], ["visible", "uncertain"])

    def test_apostrophe_split_by_transcription_is_rejoined(self) -> None:
        tracks = [FaceTrack("face-1", 0.0, 1.0, [FaceBox(0.0, 0, 0, 10, 10)])]
        turns = [SpeakerTurn("SPEAKER_00", 0.0, 1.0, 0.9)]
        words = [Word("l", 0.1, 0.2, 0.9), Word("'Italia", 0.2, 0.4, 0.9)]
        activities = [FaceActivity("face-1", 0.0, 1.0, 0.9)]
        events = associate_speakers(turns, words, tracks, activities)
        self.assertEqual(events[0].text, "l'Italia")


if __name__ == "__main__":
    unittest.main()
