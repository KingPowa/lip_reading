import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core.audio import speech
from src.core.audio.voice_onnx import ONNX_FILENAME, fbank


class VoiceOnnxTest(unittest.TestCase):
    def test_fbank_shape_and_mean_normalised(self):
        samples = np.random.default_rng(0).normal(size=16000).astype(np.float32)
        features = fbank(samples)
        self.assertEqual(features.shape, (1, 101, 80))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features.mean(axis=1), 0, atol=1e-4)

    def test_live_uses_onnx_embedder_when_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / speech.SPEAKER_MODEL_DIRECTORY
            folder.mkdir()
            (folder / ONNX_FILENAME).touch()
            fake = object()
            with patch.object(speech, "OnnxSpeakerEmbedder", return_value=fake):
                self.assertIs(speech.warm_up_speaker_model(Path(directory)), fake)

    def test_live_falls_back_to_speechbrain_when_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(speech, "_speaker_classifier", return_value="sb") as load,
                self.assertLogs(speech.LOGGER, "WARNING"),
            ):
                self.assertEqual(speech.warm_up_speaker_model(Path(directory)), "sb")
            load.assert_called_once()

    def test_embed_samples_routes_onnx_embedder(self):
        class Fake(speech.OnnxSpeakerEmbedder):
            def __init__(self):
                pass

            def __call__(self, samples):
                return np.array([3.0, 4.0])

        embedding = speech.embed_samples(Fake(), np.zeros(32000, np.float32))
        np.testing.assert_allclose(embedding, [0.6, 0.8])
