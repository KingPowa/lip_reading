import unittest

from tests.evaluate_live import evaluate


class ReplayEvaluationTest(unittest.TestCase):
    def test_only_completed_reference_sections_count_and_missing_words_are_measured(
        self,
    ):
        report = {
            "sourceTime": 10,
            "maximumConcurrent": 1,
            "drained": True,
            "startEnabled": True,
            "visual": {},
            "snapshots": [
                {"at": 9, "caption": "Person 1: uno tre"},
                {"at": 12, "caption": "Person 1: uno tre"},
            ],
            "responses": [
                {
                    "status": 200,
                    "receivedAt": 10,
                    "timings": {"total": 2},
                    "events": [
                        {
                            "start": 1,
                            "end": 4,
                            "text": "uno tre",
                            "speaker": "Person 1",
                            "faceTrack": "face",
                        }
                    ],
                }
            ],
        }
        reference = "1 secondo\nuno due tre\n0:08\n8 secondi\nnot yet completed\n0:17\n17 secondi\n"
        result = evaluate(report, reference)
        self.assertEqual(result["referenceEndSeconds"], 8)
        self.assertEqual(result["referenceWords"], 3)
        self.assertAlmostEqual(result["wordErrorRate"], 1 / 3)
        self.assertEqual(result["medianCaptionArrivalDelaySeconds"], 6)
        self.assertEqual(result["medianCaptionDisplayDelaySeconds"], 8)
        self.assertEqual(result["observedDisplayedEvents"], 1)


if __name__ == "__main__":
    unittest.main()
