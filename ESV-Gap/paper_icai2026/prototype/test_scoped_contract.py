"""Synthetic policy tests only; these are not scientific-discovery results."""
import unittest
from dataclasses import replace
from datetime import date
from scoped_contract import Evidence, Hypothesis, Requirement, evaluate


class ScopedContractTests(unittest.TestCase):
    def setUp(self):
        self.h = Hypothesis("iot_ids", frozenset({"device_A", "held_out_attack"}),
                            (Requirement("latency", "<=", 10, "ms"),
                             Requirement("recall", ">=", .90, "fraction")), date(2024, 12, 31))
        self.e = Evidence("synthetic_1", "run_A", date(2024, 1, 1), "iot_ids",
                          self.h.conditions, {"latency": (8, "ms"), "recall": (.95, "fraction")},
                          "Synthetic test fixture only.", "Synthetic test fixture only.", True)

    def test_complete_single_experiment_closes(self):
        self.assertEqual(evaluate(self.h, [self.e], search_complete=True)["disposition"],
                         "CLOSED_WITHIN_SCOPE")

    def test_partial_does_not_close(self):
        result = evaluate(self.h, [replace(self.e, conditions=frozenset({"device_A"}))],
                          search_complete=True)
        self.assertEqual(result["disposition"], "PARTIAL_EVIDENCE")
        self.assertFalse(result["novelty_established"])

    def test_separate_experiments_cannot_be_merged(self):
        a = replace(self.e, metrics={"latency": (8, "ms")})
        b = replace(self.e, experiment_id="run_B", metrics={"recall": (.95, "fraction")})
        self.assertEqual(evaluate(self.h, [a, b], search_complete=True)["disposition"],
                         "PARTIAL_EVIDENCE")

    def test_future_evidence_excluded(self):
        result = evaluate(self.h, [replace(self.e, available_on=date(2025, 1, 1))], search_complete=True)
        self.assertEqual(result["disposition"], "OPEN_FOR_REVIEW")
        self.assertEqual(result["traces"][0]["status"], "excluded_future")

    def test_failed_search_requires_review(self):
        self.assertEqual(evaluate(self.h, [], search_complete=False)["disposition"], "REVIEW_REQUIRED")

    def test_positive_closure_survives_other_search_failure(self):
        self.assertEqual(evaluate(self.h, [self.e], search_complete=False)["disposition"],
                         "CLOSED_WITHIN_SCOPE")

    def test_unchecked_or_fabricated_span_requires_review(self):
        for record in (replace(self.e, verified_span=False), replace(self.e, span="fabricated")):
            self.assertEqual(evaluate(self.h, [record], search_complete=True)["disposition"],
                             "REVIEW_REQUIRED")

    def test_unit_mismatch_is_not_a_failure_measurement(self):
        result = evaluate(self.h, [replace(self.e, metrics={"latency": (8, "s")})], search_complete=True)
        self.assertEqual(result["traces"][0]["requirements"][0]["state"], "incomparable")

    def test_missing_value_is_not_negative_evidence(self):
        result = evaluate(self.h, [replace(self.e, metrics={})], search_complete=True)
        self.assertEqual(result["traces"][0]["requirements"][0]["state"], "unreported")

    def test_bad_requirement_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(replace(self.h, requirements=(Requirement("latency", "==", 10, "ms"),)),
                     [self.e], search_complete=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
