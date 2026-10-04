import unittest

from src.core.contracts import FaceActivity, FaceBox, FaceTrack, SpeakerTurn, Word
from src.core.tracking.association import associate_speakers


class AssociationTest(unittest.TestCase):
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
