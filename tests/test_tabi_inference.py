"""Regression tests for the TABI generation contract."""

import unittest

from src.tabi_inference import _cluster_prompt_nodes, _validate_tabi_output


class TabiInferenceTests(unittest.TestCase):
    def test_prompt_uses_compatibility_representatives_in_recorded_order(self):
        cluster = {
            "nodes": ["arbitrary-a", "arbitrary-b"],
            "representative_nodes": ["top-degree", "second", "top-degree"],
        }
        self.assertEqual(_cluster_prompt_nodes(cluster), ["top-degree", "second"])

    def test_current_bucket_is_validated_and_text_is_trimmed(self):
        result = _validate_tabi_output({
            "Grounds": " graph statistic ",
            "Claim": " testable question ",
            "Warrant": " conditional rationale ",
            "Bucket": "near_term_feasible",
        })
        self.assertEqual(result["Bucket"], "near_term_feasible")
        self.assertEqual(result["Grounds"], "graph statistic")

    def test_invalid_bucket_or_missing_text_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_tabi_output({
                "Grounds": "ground", "Claim": "claim", "Warrant": "warrant",
                "Bucket": "more_probable",
            })
        with self.assertRaises(ValueError):
            _validate_tabi_output({
                "Grounds": "ground", "Claim": "", "Warrant": "warrant",
                "Bucket": "near_term_feasible",
            })


if __name__ == "__main__":
    unittest.main()
