"""Automated consistency check between canonical results, experiment artifacts, and manuscript."""

import json
from pathlib import Path
import unittest


class ResultConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.canonical_path = self.base_dir / "paper_v2" / "canonical_results.json"
        self.saturation_path = (
            self.base_dir
            / "runs"
            / "deep_learning_iot_intrusion_de_20260831_114802"
            / "outputs"
            / "corpus_saturation_report.json"
        )
        self.gold_path = (
            self.base_dir
            / "runs"
            / "deep_learning_iot_intrusion_de_20260831_114802"
            / "outputs"
            / "extractor_recall_report.json"
        )
        self.figure_path = self.base_dir / "paper_v2" / "figures" / "corpus_saturation_curve.png"

    def test_canonical_manifest_exists(self):
        self.assertTrue(self.canonical_path.exists(), "canonical_results.json must exist")
        self.assertTrue(self.saturation_path.exists(), "frozen corpus_saturation_report.json must exist")
        self.assertTrue(self.gold_path.exists(), "frozen extractor_recall_report.json must exist")
        self.assertTrue(self.figure_path.exists(), "figure corpus_saturation_curve.png must exist")

    def test_corpus_saturation_consistency(self):
        with open(self.canonical_path, encoding="utf-8") as f:
            canonical = json.load(f)
        with open(self.saturation_path, encoding="utf-8") as f:
            saturation = json.load(f)

        c_sat = canonical["corpus_saturation"]
        s_sum = saturation["summary"]

        self.assertEqual(c_sat["total_papers"], s_sum["total_papers"])
        self.assertEqual(c_sat["marginal_initial_mean"], s_sum["marginal_initial_mean"])
        self.assertEqual(c_sat["marginal_final_mean"], s_sum["marginal_final_mean"])
        self.assertEqual(c_sat["discovery_decay_rate"], s_sum["discovery_decay_rate"])
        self.assertEqual(c_sat["heaps_beta"], s_sum["heaps_law"]["beta"])
        self.assertEqual(c_sat["heaps_k"], s_sum["heaps_law"]["K"])
        self.assertEqual(c_sat["heaps_r_squared"], s_sum["heaps_law"]["r_squared"])
        self.assertEqual(c_sat["verdict"], s_sum["verdict"])

    def test_extractor_gold_benchmark_consistency(self):
        with open(self.canonical_path, encoding="utf-8") as f:
            canonical = json.load(f)
        with open(self.gold_path, encoding="utf-8") as f:
            gold = json.load(f)

        c_gold = canonical["gold_benchmark"]
        g_sum = gold["summary"]

        self.assertEqual(c_gold["gold_triples_total"], g_sum["gold_triples_total"])
        self.assertEqual(c_gold["true_positives"], g_sum["true_positives"])
        self.assertEqual(c_gold["false_negatives"], g_sum["false_negatives"])
        self.assertEqual(c_gold["false_positives"], g_sum["false_positives"])
        self.assertEqual(c_gold["recall"], g_sum["recall"])
        self.assertEqual(c_gold["miss_rate"], g_sum["miss_rate"])
        self.assertEqual(c_gold["precision"], g_sum["precision"])
        self.assertEqual(c_gold["f1_score"], g_sum["f1_score"])

    def test_candidates_consistency(self):
        with open(self.canonical_path, encoding="utf-8") as f:
            canonical = json.load(f)
        candidates = canonical["candidates"]
        self.assertEqual(len(candidates), 5)

        dispositions = [c["disposition"] for c in candidates]
        self.assertEqual(dispositions.count("REFUTED"), 3)
        self.assertEqual(dispositions.count("REVIEW_REQUIRED"), 1)
        self.assertEqual(dispositions.count("EVIDENCE_SUPPORTED_BUT_OPEN"), 1)

    def test_canonical_entities_invariant(self):
        """Invariant: V_observed(N_final) == total_unique_canonical_entities."""
        with open(self.canonical_path, encoding="utf-8") as f:
            canonical = json.load(f)
        canonical_report_path = self.base_dir / "outputs" / "corpus_saturation_report_canonical.json"
        self.assertTrue(canonical_report_path.exists(), "canonical saturation report must exist")
        with open(canonical_report_path, encoding="utf-8") as f:
            can_rep = json.load(f)

        final_observed_endpoint = can_rep["series"][-1]["cumulative_entities"]
        total_unique_canonical = canonical["multigraph"]["canonical_entities"]
        self.assertEqual(final_observed_endpoint, total_unique_canonical)
        self.assertEqual(final_observed_endpoint, 1185)

    def test_k_fetch_sensitivity_consistency(self):
        """Invariant: Under both strict and expanded protocols, verified counterevidence is 0 across all depths."""
        with open(self.canonical_path, encoding="utf-8") as f:
            canonical = json.load(f)
        k_sens = canonical.get("k_fetch_sensitivity", {})
        self.assertIn("strict_canonical_protocol", k_sens)
        self.assertIn("expanded_synonym_protocol", k_sens)

        # In strict protocol: K=5 has 0 co-mentions (matches Table 3)
        strict_k5 = k_sens["strict_canonical_protocol"]["K=5"]
        self.assertEqual(strict_k5["lexical_co_mentions"], 0)
        self.assertEqual(strict_k5["verified_counterevidence"], 0)

        # In expanded synonym protocol: K=5 has 4 co-mentions
        exp_k5 = k_sens["expanded_synonym_protocol"]["K=5"]
        self.assertEqual(exp_k5["lexical_co_mentions"], 4)
        self.assertEqual(exp_k5["verified_counterevidence"], 0)

        # In all evaluated depths, verified counterevidence is 0
        for proto in ["strict_canonical_protocol", "expanded_synonym_protocol"]:
            for k_str, res in k_sens[proto].items():
                self.assertEqual(res["verified_counterevidence"], 0, f"Counterevidence must be 0 for {proto} {k_str}")
                self.assertEqual(res["disposition"], "SUPPORTED_OPEN")


if __name__ == "__main__":
    unittest.main()


