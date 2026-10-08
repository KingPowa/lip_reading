import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.core.vision.face_identity import FaceIdentityEncoder


class FaceIdentityEncoderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_directory = Path(self.directory.name)
        self.model_path = self.model_directory / "face_recognition_sface_2021dec.onnx"
        self.content = b"test model"
        self.checksum = hashlib.sha256(self.content).hexdigest()
        self.checksum_patch = patch(
            "src.core.vision.face_identity.FACE_IDENTITY_MODEL_SHA256", self.checksum
        )
        self.checksum_patch.start()
        self.addCleanup(self.checksum_patch.stop)
        self.network = Mock()
        self.network.feature.return_value = np.ones((1, 128), dtype=np.float32)
        self.network_patch = patch(
            "src.core.vision.face_identity.cv2.FaceRecognizerSF.create",
            return_value=self.network,
        )
        self.create_model = self.network_patch.start()
        self.addCleanup(self.network_patch.stop)

    def encoder(self):
        self.model_path.write_bytes(self.content)
        return FaceIdentityEncoder(self.model_directory)

    def landmarks(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
        for index in (33, 133):
            points[index] = SimpleNamespace(x=0.3, y=0.35)
        for index in (362, 263):
            points[index] = SimpleNamespace(x=0.7, y=0.35)
        points[1] = SimpleNamespace(x=0.5, y=0.55)
        points[61] = SimpleNamespace(x=0.35, y=0.75)
        points[291] = SimpleNamespace(x=0.65, y=0.75)
        return points

    def test_aligns_landmarks_in_crop_coordinates_and_returns_unit_features(self):
        encoder = self.encoder()
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        features = encoder.encode(image, self.landmarks())
        alignment = self.network.alignCrop.call_args.args[1]
        np.testing.assert_allclose(
            alignment[4:14], [30, 70, 70, 70, 50, 110, 35, 150, 65, 150]
        )
        self.assertEqual(features.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(features)), 1.0, places=6)

    def test_missing_or_degenerate_landmarks_do_not_produce_features(self):
        encoder = self.encoder()
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        self.assertIsNone(encoder.encode(image, []))
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
        self.assertIsNone(encoder.encode(image, points))
        points = self.landmarks()
        points[1].x = float("nan")
        self.assertIsNone(encoder.encode(image, points))
        self.network.alignCrop.assert_not_called()

    def test_invalid_network_features_do_not_enter_tracking(self):
        encoder = self.encoder()
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        for features in (np.zeros((1, 128)), np.full((1, 128), np.nan)):
            self.network.feature.return_value = features
            self.assertIsNone(encoder.encode(image, self.landmarks()))

    def test_missing_model_download_is_verified_and_atomically_saved(self):
        with patch(
            "src.core.vision.face_identity.urllib.request.urlopen",
            return_value=BytesIO(self.content),
        ) as download:
            FaceIdentityEncoder(self.model_directory)
        download.assert_called_once()
        self.assertEqual(self.model_path.read_bytes(), self.content)
        self.assertEqual(list(self.model_directory.iterdir()), [self.model_path])

    def test_existing_model_is_reused_without_network(self):
        with patch("src.core.vision.face_identity.urllib.request.urlopen") as download:
            self.encoder()
        download.assert_not_called()

    def test_corrupt_model_is_rejected_before_native_loading(self):
        self.model_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            FaceIdentityEncoder(self.model_directory)
        self.create_model.assert_not_called()

    def test_bad_download_and_network_failure_leave_no_partial_model(self):
        for response in (BytesIO(b"corrupt"), OSError("unavailable")):
            options = {"return_value": response}
            if isinstance(response, OSError):
                options = {"side_effect": response}
            with (
                patch(
                    "src.core.vision.face_identity.urllib.request.urlopen", **options
                ),
                self.assertRaises(RuntimeError),
            ):
                FaceIdentityEncoder(self.model_directory)
            self.assertEqual(list(self.model_directory.iterdir()), [])
