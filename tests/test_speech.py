import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.core.audio.speech import (
    FasterWhisperAdapter,
    transcribe_speech,
    uncommitted_words,
)
from src.core.contracts import Word


class TranscriptionTest(unittest.TestCase):
    def test_normalized_pcm_goes_directly_to_whisper_without_another_ffmpeg_process(
        self,
    ):
        class Model:
            def transcribe(self, audio, **options):
                self.audio = audio
                return {"segments": []}

        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "audio.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(
                    np.array([-32768, 0, 32767], dtype=np.int16).tobytes()
                )
            transcribe_speech(audio_path, "base", Path("models"), model=model)
        self.assertEqual(model.audio.dtype, np.float32)
        np.testing.assert_allclose(model.audio, [-32768 / 32767, 0, 1])

    def test_overlap_text_preserves_words_with_revised_timestamps(self) -> None:
        words = [
            Word("realtà", 0.0, 0.4, 0.9),
            Word("mitologica.", 0.4, 0.7, 0.9),
            Word("Io", 0.7, 0.9, 0.9),
            Word("credo", 0.9, 1.1, 0.9),
            Word("tail", 2.8, 3.4, 0.9),
        ]
        result = uncommitted_words(words, "una realtà mitologica.", 1.0, 3.0)
        self.assertEqual([word.text for word in result], ["Io", "credo"])

    def test_unmatched_overlap_falls_back_to_time_without_dropping_repeated_words(
        self,
    ) -> None:
        words = [Word("no", 0.1, 0.3, 0.9), Word("no", 1.1, 1.3, 0.9)]
        result = uncommitted_words(words, "no", 1.0, 2.0)
        self.assertEqual([word.text for word in result], ["no"])

    def test_repeated_phrase_after_the_committed_anchor_is_retained(self) -> None:
        words = [
            Word(text, index * 0.2, (index + 1) * 0.2, 0.9)
            for index, text in enumerate(
                "ci metto la faccia ci metto la faccia".split()
            )
        ]
        result = uncommitted_words(words, "ci metto la faccia", 0.8, 2.0)
        self.assertEqual(
            [word.text for word in result], ["ci", "metto", "la", "faccia"]
        )

    def test_transcription_uses_the_selected_language_without_translation(self) -> None:
        class Model:
            def transcribe(self, *_arguments, **options):
                self.options = options
                return {"segments": []}

        model = Model()
        samples = np.zeros(16000, dtype=np.float32)
        with patch(
            "src.core.audio.speech._audio_samples", return_value=(samples, 16000)
        ):
            transcribe_speech(Path("audio.wav"), "base", Path("models"), "IT", model)
        self.assertEqual(model.options["language"], "it")
        self.assertEqual(model.options["task"], "transcribe")

    def test_live_transcription_can_skip_word_alignment(self) -> None:
        class Model:
            def transcribe(self, *_arguments, **options):
                self.options = options
                return {"segments": []}

        model = Model()
        with patch(
            "src.core.audio.speech._audio_samples",
            return_value=(np.zeros(16000, dtype=np.float32), 16000),
        ):
            transcribe_speech(
                Path("audio.wav"),
                "base",
                Path("models"),
                model=model,
                word_timestamps=False,
            )
        self.assertFalse(model.options["word_timestamps"])

    def test_faster_whisper_segments_use_the_shared_timestamped_word_contract(self):
        class Model:
            def transcribe(self, _audio, **options):
                self.options = options
                word = SimpleNamespace(
                    word=" Ciao", start=0.1, end=0.4, probability=0.95
                )
                segment = SimpleNamespace(
                    start=0.1,
                    end=0.4,
                    text=" Ciao",
                    avg_logprob=-0.1,
                    words=[word],
                )
                return iter([segment]), SimpleNamespace(language="it")

        model = Model()
        adapter = FasterWhisperAdapter(model)
        with patch(
            "src.core.audio.speech._audio_samples",
            return_value=(np.zeros(16000, dtype=np.float32), 16000),
        ):
            turns, words = transcribe_speech(
                Path("audio.wav"), "base", Path("models"), "IT", adapter
            )
        self.assertEqual(model.options["language"], "it")
        self.assertFalse(model.options["condition_on_previous_text"])
        self.assertNotIn("initial_prompt", model.options)
        self.assertEqual(words[0], Word("Ciao", 0.1, 0.4, 0.95))
        self.assertEqual((turns[0].start, turns[0].end), (0.1, 0.4))


if __name__ == "__main__":
    unittest.main()


class VoiceEmbeddingTest(unittest.TestCase):
    class Classifier:
        def encode_batch(self, signal):
            import torch

            self.shape = tuple(signal.shape)
            return torch.tensor([[[3.0, 4.0]]])

    def test_embed_samples_returns_a_unit_vector(self):
        from src.core.audio.speech import embed_samples

        classifier = self.Classifier()
        embedding = embed_samples(classifier, np.zeros(16_000, dtype=np.float32))
        np.testing.assert_allclose(embedding, [0.6, 0.8], rtol=1e-6)
        self.assertEqual(classifier.shape, (1, 16_000))

    def test_embed_samples_rejects_segments_shorter_than_the_minimum(self):
        from src.core.audio.speech import embed_samples

        self.assertIsNone(
            embed_samples(self.Classifier(), np.zeros(1_000, dtype=np.float32))
        )

    def test_offline_turn_embedding_uses_the_same_embedding(self):
        from src.core.audio.speech import _turn_embedding
        from src.core.contracts import SpeakerTurn

        embedding = _turn_embedding(
            self.Classifier(),
            np.zeros(32_000, dtype=np.float32),
            16_000,
            SpeakerTurn("a", 0.0, 1.0, 0.9),
        )
        np.testing.assert_allclose(embedding, [0.6, 0.8], rtol=1e-6)
