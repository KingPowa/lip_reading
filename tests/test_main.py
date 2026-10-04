import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.app.offline import recognize
from src.core.audio.speech import diarize_turns
from src.core.contracts import FaceActivity, FaceBox, FaceTrack, SpeakerTurn, Word
from src.core.media import MediaInfo


class LiveVoiceIdentityTest(unittest.TestCase):
    def test_two_visible_speakers_inside_one_whisper_segment_have_distinct_voices(
        self,
    ) -> None:
        tracks = [
            FaceTrack("first-face", 0.0, 1.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("second-face", 1.0, 2.0, [FaceBox(1.0, 10, 0, 10, 10)]),
        ]
        with (
            patch(
                "src.app.offline.inspect_media",
                return_value=MediaInfo(2.0, 20, 10, 25.0, True, True),
            ),
            patch("src.app.offline.read_video_frames", return_value=[]),
            patch(
                "src.app.offline.transcribe_speech",
                return_value=(
                    [SpeakerTurn("one-whisper-segment", 0.0, 2.0, 0.9)],
                    [Word("first", 0.0, 0.9, 0.9), Word("second", 1.0, 1.9, 0.9)],
                ),
            ),
            patch("src.app.offline.analyze_faces", return_value=tracks),
            patch(
                "src.app.offline.score_face_activity",
                return_value=[
                    FaceActivity("first-face", 0.0, 1.0, 0.9),
                    FaceActivity("second-face", 1.0, 2.0, 0.9),
                ],
            ),
            patch(
                "src.core.audio.speech._audio_samples",
                return_value=(np.zeros(32_000), 16_000),
            ),
            patch("src.core.audio.speech._speaker_classifier", return_value=object()),
            patch(
                "src.core.audio.speech._turn_embedding",
                side_effect=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            ),
        ):
            _, _, events, _, _ = recognize(
                Path("clip.mp4"),
                Path("audio.wav"),
                live=True,
                speaker_centroids=[],
            )
        self.assertEqual([event.speaker for event in events], ["Person 1", "Person 2"])
        self.assertEqual([event.text for event in events], ["first", "second"])

    def test_short_live_reply_keeps_its_visual_identity_without_an_embedding(
        self,
    ) -> None:
        for preserve_visual_identity, expected_reply in (
            (True, "reply-face"),
            (False, "SPEAKER_00"),
        ):
            with (
                self.subTest(preserve_visual_identity=preserve_visual_identity),
                patch(
                    "src.core.audio.speech._audio_samples",
                    return_value=(np.zeros(32_000), 16_000),
                ),
                patch(
                    "src.core.audio.speech._speaker_classifier", return_value=object()
                ),
                patch(
                    "src.core.audio.speech._turn_embedding",
                    side_effect=[np.array([1.0, 0.0]), None],
                ),
            ):
                turns = diarize_turns(
                    Path("audio.wav"),
                    [
                        SpeakerTurn("first-face", 0.0, 1.0, 0.9),
                        SpeakerTurn("reply-face", 1.0, 1.2, 0.9),
                    ],
                    Path("models"),
                    [],
                    preserve_short_turn_ids=preserve_visual_identity,
                )
                self.assertEqual(
                    [turn.speaker_id for turn in turns], ["SPEAKER_00", expected_reply]
                )

    def test_live_voice_clustering_preserves_unassigned_speech(self) -> None:
        visible_tracks = [
            FaceTrack("first-face", 0.0, 1.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("second-face", 0.0, 1.0, [FaceBox(0.0, 10, 0, 10, 10)]),
        ]
        for tracks, activities, expected_state in (
            ([], [], "off_screen"),
            (
                visible_tracks,
                [
                    FaceActivity("first-face", 0.0, 1.0, 0.75),
                    FaceActivity("second-face", 0.0, 1.0, 0.70),
                ],
                "uncertain",
            ),
        ):
            with (
                self.subTest(expected_state=expected_state),
                patch(
                    "src.app.offline.inspect_media",
                    return_value=MediaInfo(1.0, 20, 10, 25.0, True, True),
                ),
                patch("src.app.offline.read_video_frames", return_value=[]),
                patch(
                    "src.app.offline.transcribe_speech",
                    return_value=(
                        [SpeakerTurn("voice", 0.0, 1.0, 0.9)],
                        [Word("hello", 0.0, 0.9, 0.9)],
                    ),
                ),
                patch("src.app.offline.analyze_faces", return_value=tracks),
                patch("src.app.offline.score_face_activity", return_value=activities),
                patch(
                    "src.core.audio.speech._audio_samples",
                    return_value=(np.zeros(16_000), 16_000),
                ),
                patch(
                    "src.core.audio.speech._speaker_classifier", return_value=object()
                ),
                patch(
                    "src.core.audio.speech._turn_embedding",
                    return_value=np.array([1.0, 0.0]),
                ),
            ):
                _, _, events, _, _ = recognize(
                    Path("clip.mp4"), Path("audio.wav"), live=True, speaker_centroids=[]
                )
                self.assertEqual([event.text for event in events], ["hello"])
                self.assertEqual([event.state for event in events], [expected_state])
                self.assertIsNone(events[0].face_track)

    def test_live_commit_excludes_old_words_and_words_crossing_the_tail(self) -> None:
        words = [
            Word("old", 0.2, 0.5, 0.9),
            Word("new", 1.3, 1.8, 0.9),
            Word("tail", 2.8, 3.4, 0.9),
        ]
        with (
            patch(
                "src.app.offline.inspect_media",
                return_value=MediaInfo(4.0, 10, 10, 25.0, True, True),
            ),
            patch("src.app.offline.read_video_frames", return_value=[]),
            patch(
                "src.app.offline.transcribe_speech",
                return_value=([SpeakerTurn("voice", 0, 4, 0.9)], words),
            ),
            patch("src.app.offline.analyze_faces", return_value=[]),
            patch("src.app.offline.score_face_activity", return_value=[]),
        ):
            _, _, events, _, _ = recognize(
                Path("clip.mp4"),
                Path("audio.wav"),
                live=True,
                minimum_word_timestamp=1.0,
                maximum_word_timestamp=3.0,
            )
        self.assertEqual([event.text for event in events], ["new"])

    def test_live_voice_retains_person_label_when_the_face_track_changes(self) -> None:
        centroids = []
        voice_labels = {}
        track_labels = {}
        labels = []
        for track_id in ("face-before-movement", "face-after-movement"):
            track = FaceTrack(track_id, 0.0, 1.0, [FaceBox(0.0, 0, 0, 10, 10)])
            with (
                patch(
                    "src.app.offline.inspect_media",
                    return_value=MediaInfo(1.0, 10, 10, 25.0, True, True),
                ),
                patch("src.app.offline.read_video_frames", return_value=[]),
                patch(
                    "src.app.offline.transcribe_speech",
                    return_value=(
                        [SpeakerTurn("initial-voice", 0.0, 1.0, 0.9)],
                        [Word("hello", 0.1, 0.8, 0.9)],
                    ),
                ),
                patch("src.app.offline.analyze_faces", return_value=[track]),
                patch(
                    "src.app.offline.score_face_activity",
                    return_value=[FaceActivity(track_id, 0.0, 1.0, 0.9)],
                ),
                patch(
                    "src.core.audio.speech._audio_samples",
                    return_value=(np.zeros(16_000), 16_000),
                ),
                patch(
                    "src.core.audio.speech._speaker_classifier", return_value=object()
                ),
                patch(
                    "src.core.audio.speech._turn_embedding",
                    return_value=np.array([1.0, 0.0]),
                ),
            ):
                _, _, events, _, _ = recognize(
                    Path("recording.mp4"),
                    Path("audio.wav"),
                    live=True,
                    speaker_centroids=centroids,
                    voice_labels=voice_labels,
                    track_labels=track_labels,
                )
            labels.append(events[0].speaker)
        self.assertEqual(labels, ["Person 1", "Person 1"])
        self.assertEqual(len(centroids), 1)

    def test_live_words_keep_the_voice_of_their_diarized_segment(self) -> None:
        words = [
            Word("hello", 0.1, 0.3, 0.9),
            Word("again", 0.35, 0.45, 0.9),
            Word("another", 1.1, 1.3, 0.9),
            Word("voice", 1.35, 1.5, 0.9),
        ]
        tracks = [
            FaceTrack("face-1", 0.0, 1.0, [FaceBox(0.0, 0, 0, 10, 10)]),
            FaceTrack("face-2", 1.0, 2.0, [FaceBox(1.0, 10, 0, 10, 10)]),
        ]
        with (
            patch(
                "src.app.offline.inspect_media",
                return_value=MediaInfo(2.0, 20, 10, 25.0, True, True),
            ),
            patch("src.app.offline.read_video_frames", return_value=[]),
            patch(
                "src.app.offline.transcribe_speech",
                return_value=(
                    [
                        SpeakerTurn("initial-voice", 0.0, 1.0, 0.9),
                        SpeakerTurn("initial-voice", 1.0, 2.0, 0.9),
                    ],
                    words,
                ),
            ),
            patch("src.app.offline.analyze_faces", return_value=tracks),
            patch(
                "src.app.offline.score_face_activity",
                return_value=[
                    FaceActivity("face-1", 0.0, 1.0, 0.9),
                    FaceActivity("face-2", 1.0, 2.0, 0.9),
                ],
            ),
            patch(
                "src.core.audio.speech._audio_samples",
                return_value=(np.zeros(32_000), 16_000),
            ),
            patch("src.core.audio.speech._speaker_classifier", return_value=object()),
            patch(
                "src.core.audio.speech._turn_embedding",
                side_effect=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            ),
        ):
            _, _, events, turns, _ = recognize(
                Path("recording.mp4"),
                Path("audio.wav"),
                live=True,
                speaker_centroids=[],
                voice_labels={
                    "SPEAKER_00": "Known speaker",
                    "SPEAKER_01": "Other speaker",
                },
            )
        self.assertEqual(
            [turn.speaker_id for turn in turns], ["SPEAKER_00", "SPEAKER_01"]
        )
        self.assertEqual(
            [event.speaker for event in events], ["Known speaker", "Other speaker"]
        )
        self.assertEqual(
            [event.text for event in events], ["hello again", "another voice"]
        )


if __name__ == "__main__":
    unittest.main()
