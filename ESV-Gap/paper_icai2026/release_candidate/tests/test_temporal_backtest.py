import unittest

from src.temporal_backtest import (
    build_temporal_graph,
    candidate_matches_control,
    evaluate_ranked_candidates,
    limitation_resolution_controls,
    structural_future_controls,
)


def triple(subject, subject_type, relation, obj, object_type, paper, year, evidence):
    return {
        "subject": {"name": subject, "type": subject_type},
        "relation": relation,
        "object": {"name": obj, "type": object_type},
        "source_paper_id": paper,
        "source_year": year,
        "confidence": 0.9,
        "evidence": evidence,
    }


class TemporalBacktestTests(unittest.TestCase):
    def test_limitation_requires_same_paper_solution_action(self):
        records = [
            triple("legacy IDS", "METHOD", "LACKS", "zero-day robustness", "CONCEPT", "future", 2025,
                   "Legacy IDS lacks zero-day robustness."),
            triple("adaptive IDS", "METHOD", "ADDRESSES", "zero-day robustness", "CONCEPT", "future", 2025,
                   "We propose and evaluate an adaptive model to address zero-day robustness."),
        ]
        controls = limitation_resolution_controls(records, start_year=2024, label="positive")
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["label"], "positive")

    def test_limitation_without_action_is_not_forced_negative(self):
        records = [
            triple("legacy IDS", "METHOD", "LACKS", "zero-day robustness", "CONCEPT", "future", 2025,
                   "Legacy IDS lacks zero-day robustness."),
        ]
        self.assertEqual(limitation_resolution_controls(records, start_year=2024, label="positive"), [])

    def test_future_typed_relation_requires_pre_cutoff_marginal_support(self):
        pre = [
            triple("adaptive IDS", "METHOD", "USES", "CNN", "METHOD", "p1", 2022, "uses CNN"),
            triple("adaptive IDS", "METHOD", "USES", "LSTM", "METHOD", "p2", 2022, "uses LSTM"),
            triple("IoT-23", "DATASET", "EVALUATES_ON", "benchmark task", "CONCEPT", "p3", 2022, "benchmark"),
            triple("IoT-23", "DATASET", "EVALUATES_ON", "attack traffic", "CONCEPT", "p4", 2022, "benchmark"),
        ]
        future = [
            triple("adaptive IDS", "METHOD", "EVALUATES_ON", "IoT-23", "DATASET", "p5", 2025,
                   "We evaluate adaptive IDS on IoT-23."),
        ]
        controls = structural_future_controls(build_temporal_graph(pre), future, 2)
        self.assertEqual(len(controls), 1)

    def test_ranked_metrics_keep_unlabelled_as_abstentions(self):
        positive = {
            "control_id": "pos", "control_type": "limitation_resolution",
            "missing_capability": "zero-day robustness", "subject": "IDS",
        }
        negative = {
            "control_id": "neg", "control_type": "limitation_resolution",
            "missing_capability": "explainability", "subject": "IDS",
        }
        candidates = [
            {"type": "evidence_gap", "subject": "IDS", "missing_capability": "zero-day robustness",
             "mean_evidence_confidence": 0.9},
            {"type": "evidence_gap", "subject": "IDS", "missing_capability": "open-set calibration",
             "mean_evidence_confidence": 0.8},
        ]
        self.assertTrue(candidate_matches_control(candidates[0], positive))
        metrics = evaluate_ranked_candidates(candidates, [positive], [negative], ks=(1, 2))
        self.assertEqual(metrics["candidate_recall_at_k"]["1"], 1.0)
        self.assertEqual(metrics["label_abstention_rate"], 0.5)
        self.assertIsNone(metrics["certificate_precision"])


if __name__ == "__main__":
    unittest.main()

