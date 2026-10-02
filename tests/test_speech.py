import unittest
from pathlib import Path

from src.speech import transcribe_speech


class TranscriptionTest(unittest.TestCase):
    def test_transcription_uses_the_selected_language_without_translation(self) -> None:
        class Model:
            def transcribe(self, *_arguments, **options):
                self.options = options
                return {"segments": []}

        model = Model()
        transcribe_speech(Path("audio.wav"), "base", Path("models"), "IT", model)
        self.assertEqual(model.options["language"], "it")
        self.assertEqual(model.options["task"], "transcribe")

    def test_live_transcription_can_skip_word_alignment(self) -> None:
        class Model:
            def transcribe(self, *_arguments, **options):
                self.options = options
                return {"segments": []}

        model = Model()
        transcribe_speech(Path("audio.wav"), "base", Path("models"), model=model, word_timestamps=False)
        self.assertFalse(model.options["word_timestamps"])

if __name__ == "__main__":
    unittest.main()
