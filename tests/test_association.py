import unittest

from src.association import associate_speakers
from src.contracts import FaceActivity, FaceTrack, FaceBox, SpeakerTurn, Word


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
        activities = [FaceActivity("face-1", 0.2, 1.0, 0.71), FaceActivity("face-2", 0.2, 1.0, 0.70)]
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
        self.assertEqual([event.state for event in events], ["overlapping", "overlapping"])


if __name__ == "__main__":
    unittest.main()
