import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.person_segmentation import (
    PERSON_MODEL_FILENAME,
    PersonSegmenter,
    decode_person_masks,
    letterbox,
)


class PersonSegmentationTest(unittest.TestCase):
    def test_corrupt_weights_are_rejected_before_loading_native_network(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / PERSON_MODEL_FILENAME).write_bytes(b"corrupt weights")
            with patch("src.person_segmentation.cv2.dnn.readNetFromONNX") as loader:
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    PersonSegmenter(model_directory)
            loader.assert_not_called()

    def test_letterbox_keeps_aspect_ratio_and_records_padding(self):
        image, bounds = letterbox(np.zeros((50, 100, 3), dtype=np.uint8))
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertEqual(bounds, (0, 160, 640, 320))
        self.assertTrue(np.all(image[:160] == 114))

    def test_person_masks_remove_duplicates_nonpeople_padding_and_keep_real_shape(self):
        detections = np.zeros((1, 116, 3), dtype=np.float32)
        detections[0, :4, :] = np.array([[320], [320], [320], [320]])
        detections[0, 4, :2] = [0.9, 0.8]
        detections[0, 9, 2] = 0.95
        detections[0, 84, :] = 1
        prototypes = np.zeros((1, 32, 160, 160), dtype=np.float32)
        prototypes[0, 0] = -2
        prototypes[0, 0, 40:120, 40:80] = 2
        prototypes[0, 0, 80:120, 80:120] = 2
        masks = decode_person_masks(
            [detections, prototypes], (50, 100), (0, 160, 640, 320)
        )
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].shape, (50, 100))
        self.assertTrue(masks[0][5, 30])
        self.assertFalse(masks[0][5, 60])
        self.assertTrue(masks[0][40, 60])
        self.assertFalse(masks[0][5, 5])

    def test_no_confident_person_produces_no_mask(self):
        masks = decode_person_masks(
            [
                np.zeros((1, 116, 1), dtype=np.float32),
                np.zeros((1, 32, 160, 160), dtype=np.float32),
            ],
            (50, 100),
            (0, 160, 640, 320),
        )
        self.assertEqual(masks, [])


if __name__ == "__main__":
    unittest.main()
