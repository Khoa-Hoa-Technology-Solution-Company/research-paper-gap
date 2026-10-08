"""Constructed policy counterexamples, not human-labelled research evidence."""
import copy
import unittest

from src.scoped_verification import evaluate_scope


def fixture():
    contract = {"task": "detect unknown attacks", "domain": "IoT IDS",
                "cutoff": "2024-12-31", "scope_reviewed": True,
                "conditions": ["edge hardware"],
                "requirements": [
                    {"name": "f1", "op": ">=", "bound": 0.9, "unit": "fraction"},
                    {"name": "latency", "op": "<=", "bound": 10, "unit": "ms"}]}
    source = "IoT IDS detects unknown attacks on edge hardware; F1 0.92; latency 8 ms."
    def annotation(value, span):
        return {"value": value, "span": span, "reviewed": True}
    record = {"paper_id": "synthetic-paper", "experiment_id": "experiment-A",
              "available_on": "2024-01-01", "metadata_reviewed": True,
              "experiment_reviewed": True, "source_text": source,
              "facets": {"task": annotation(contract["task"], "detects unknown attacks"),
                         "domain": annotation(contract["domain"], "IoT IDS")},
              "conditions": {"edge hardware": annotation(True, "on edge hardware")},
              "measurements": {
                  "f1": {**annotation(0.92, "F1 0.92"), "unit": "fraction"},
                  "latency": {**annotation(8, "latency 8 ms"), "unit": "ms"}}}
    ledger = {"cutoff": contract["cutoff"], "required_probe_ids": ["q1"],
              "probes": [{"probe_id": "q1", "query": "synthetic query",
                          "provider": "synthetic", "snapshot_id": "fixture-1",
                          "status": "completed", "pagination_complete": True,
                          "semantic_review_complete": True}]}
    return contract, [record], ledger


class ScopedVerificationTests(unittest.TestCase):
    def evaluate(self, fixture_data):
        result = evaluate_scope(*fixture_data)
        self.assertIs(result["novelty_established"], False)
        self.assertEqual(len(result["input_sha256"]), 64)
        return result["disposition"]

    def test_one_complete_experiment_closes(self):
        self.assertEqual(self.evaluate(fixture()), "CLOSED_WITHIN_SCOPE")

    def test_no_union_across_experiments(self):
        c, rows, ledger = fixture()
        second = copy.deepcopy(rows[0])
        second["experiment_id"] = "experiment-B"
        rows[0]["measurements"].pop("latency")
        second["measurements"].pop("f1")
        self.assertEqual(self.evaluate((c, rows + [second], ledger)), "PARTIAL_EVIDENCE")

    def test_no_union_across_papers(self):
        c, rows, ledger = fixture()
        second = copy.deepcopy(rows[0])
        second["paper_id"] = "other-paper"
        rows[0]["measurements"].pop("latency")
        second["measurements"].pop("f1")
        self.assertEqual(self.evaluate((c, rows + [second], ledger)), "PARTIAL_EVIDENCE")

    def test_future_evidence_excluded(self):
        data = fixture(); data[1][0]["available_on"] = "2025-01-01"
        self.assertEqual(self.evaluate(data), "OPEN_FOR_REVIEW")

    def test_missing_date_requires_review(self):
        data = fixture(); data[1][0].pop("available_on")
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_metadata_not_verified(self):
        data = fixture(); data[1][0]["metadata_reviewed"] = False
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_quote_not_in_source(self):
        data = fixture(); data[1][0]["measurements"]["latency"]["span"] = "invented 8 ms"
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_quote_without_semantic_review(self):
        data = fixture(); data[1][0]["measurements"]["latency"]["reviewed"] = False
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_wrong_task_is_not_closure(self):
        data = fixture(); data[1][0]["facets"]["task"]["value"] = "detect known attacks"
        self.assertEqual(self.evaluate(data), "OPEN_FOR_REVIEW")

    def test_wrong_domain_is_not_closure(self):
        data = fixture(); data[1][0]["facets"]["domain"]["value"] = "medical imaging"
        self.assertEqual(self.evaluate(data), "OPEN_FOR_REVIEW")

    def test_missing_condition_requires_review(self):
        data = fixture(); data[1][0]["conditions"] = {}
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_condition_not_met(self):
        data = fixture(); data[1][0]["conditions"]["edge hardware"]["value"] = False
        self.assertEqual(self.evaluate(data), "PARTIAL_EVIDENCE")

    def test_unit_mismatch_is_not_silently_converted(self):
        data = fixture(); data[1][0]["measurements"]["latency"]["unit"] = "seconds"
        self.assertEqual(self.evaluate(data), "PARTIAL_EVIDENCE")

    def test_nonfinite_measurement_never_closes(self):
        data = fixture(); data[1][0]["measurements"]["latency"]["value"] = float("nan")
        self.assertEqual(self.evaluate(data), "PARTIAL_EVIDENCE")

    def test_failed_threshold_is_partial(self):
        data = fixture(); data[1][0]["measurements"]["f1"]["value"] = 0.5
        self.assertEqual(self.evaluate(data), "PARTIAL_EVIDENCE")

    def test_underspecified_contract_requires_review(self):
        data = fixture(); data[0]["conditions"] = []
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_unreviewed_contract_requires_review(self):
        data = fixture(); data[0]["scope_reviewed"] = False
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_missing_probe_never_establishes_absence(self):
        c, _, ledger = fixture(); ledger["probes"] = []
        self.assertEqual(self.evaluate((c, [], ledger)), "REVIEW_REQUIRED")

    def test_pagination_failure_never_establishes_absence(self):
        c, _, ledger = fixture(); ledger["probes"][0]["pagination_complete"] = False
        self.assertEqual(self.evaluate((c, [], ledger)), "REVIEW_REQUIRED")

    def test_empty_search_is_only_open_for_review(self):
        c, _, ledger = fixture()
        self.assertEqual(self.evaluate((c, [], ledger)), "OPEN_FOR_REVIEW")

    def test_witness_survives_other_probe_failure(self):
        data = fixture(); data[2]["probes"][0]["status"] = "timeout"
        self.assertEqual(self.evaluate(data), "CLOSED_WITHIN_SCOPE")

    def test_mutation_does_not_change_inputs(self):
        data = fixture(); original = copy.deepcopy(data)
        self.evaluate(data)
        self.assertEqual(data, original)

    def test_experiment_membership_requires_review(self):
        data = fixture(); data[1][0]["experiment_reviewed"] = False
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")

    def test_duplicate_experiment_does_not_choose_first_record(self):
        c, rows, ledger = fixture()
        second = copy.deepcopy(rows[0])
        second["measurements"]["latency"]["value"] = 100
        self.assertEqual(self.evaluate((c, rows + [second], ledger)), "REVIEW_REQUIRED")

    def test_malformed_metric_name_requires_review(self):
        data = fixture(); data[0]["requirements"][0]["name"] = ["f1"]
        self.assertEqual(self.evaluate(data), "REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
