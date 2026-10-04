import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.core.audio.models import extract_model
from src.core.audio.realtime import StreamingTranscriber, decode_pcm


class StreamingAudioTest(unittest.TestCase):
    def test_pcm_validation_rejects_empty_unaligned_and_oversized_audio(self):
        for content in (b"", b"x", b"x" * 32002):
            with self.assertRaises(ValueError):
                decode_pcm(content)
        np.testing.assert_allclose(
            decode_pcm(np.array([-32767, 0, 32767], dtype="<i2").tobytes()), [-1, 0, 1]
        )

    def test_pcm_is_fed_once_and_partial_text_does_not_require_word_alignment(self):
        decoder = Mock()
        decoder.AcceptWaveform.return_value = False
        decoder.PartialResult.return_value = '{"partial":"ciao"}'
        with patch("vosk.KaldiRecognizer", return_value=decoder):
            transcriber = StreamingTranscriber(object(), 1.0)
        update = transcriber.accept(b"pcm", False)
        decoder.AcceptWaveform.assert_called_once_with(b"pcm")
        self.assertEqual(update.partial_text, "ciao")
        self.assertEqual(update.words, [])
        decoder.SetWords.assert_called_once_with(True)

    def test_final_words_preserve_capture_clock_and_empty_flush_preserves_tail(self):
        decoder = Mock()
        decoder.FinalResult.return_value = (
            '{"result":[{"word":"ciao","start":0.1,"end":0.4,"conf":0.95}]}'
        )
        with patch("vosk.KaldiRecognizer", return_value=decoder):
            transcriber = StreamingTranscriber(object(), 1.0)
        update = transcriber.accept(b"", True)
        decoder.AcceptWaveform.assert_not_called()
        self.assertEqual(update.words[0].start, 1.1)
        self.assertEqual(update.words[0].end, 1.4)
        self.assertEqual(update.partial_text, "")

    def test_model_archive_cannot_write_outside_its_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "model.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("model/../../outside.txt", "invalid")
            with self.assertRaises(ValueError):
                extract_model(archive, directory, "model")
            self.assertFalse((directory.parent / "outside.txt").exists())
