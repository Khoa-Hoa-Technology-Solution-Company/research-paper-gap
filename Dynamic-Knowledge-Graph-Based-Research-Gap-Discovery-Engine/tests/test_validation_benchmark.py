import unittest

from experiments.run_validation_benchmark import BINARY_MAPPING, binary_prediction


class BinaryMappingTests(unittest.TestCase):
    def test_only_automatically_eligible_is_predicted_positive(self):
        self.assertEqual(binary_prediction({"status": "automatically_eligible"}), 1)
        self.assertEqual(binary_prediction({"status": "review_required"}), 0)
        self.assertEqual(binary_prediction({"status": "rejected"}), 0)

    def test_mapping_documents_both_ground_truth_classes(self):
        self.assertIn("label == 1", BINARY_MAPPING["ground_truth_positive"])
        self.assertIn("label == 0", BINARY_MAPPING["ground_truth_negative"])


if __name__ == "__main__":
    unittest.main()
