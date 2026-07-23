"""Focused tests for the appendix reporting calculations."""

from __future__ import annotations

import unittest

import networkx as nx

from src.benchmark_reporting import (
    _normalized_cut,
    _random_score_expectation,
    _tie_aware_pairwise_auc,
    _tie_aware_top1,
)


class BenchmarkReportingTests(unittest.TestCase):
    def test_pairwise_auc_awards_half_credit_for_ties(self) -> None:
        # Positive 0.5 beats 0.4, ties 0.5, and loses to 0.6: (1 + .5 + 0) / 3.
        score = _tie_aware_pairwise_auc(
            [(0.5, True), (0.4, False), (0.5, False), (0.6, False)]
        )
        self.assertAlmostEqual(score, 0.5)

    def test_top1_splits_credit_among_tied_top_candidates(self) -> None:
        self.assertAlmostEqual(
            _tie_aware_top1([(0.8, True), (0.8, False), (0.3, False)]), 0.5
        )
        self.assertEqual(_tie_aware_top1([(0.7, False), (0.6, True)]), 0.0)

    def test_normalized_cut_uses_both_set_volumes(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        # cut=1, vol({a,b})=3, vol({c,d})=3, hence Ncut=2/3.
        self.assertAlmostEqual(_normalized_cut(graph, frozenset({"a", "b"})), 2 / 3)

    def test_random_score_reference_is_exact_chance_not_seeded_draw(self) -> None:
        self.assertEqual(
            _random_score_expectation(3),
            {"conditional_pairwise_auc": 0.5, "tie_aware_top1_rate": 1 / 3},
        )
        with self.assertRaises(ValueError):
            _random_score_expectation(1)


if __name__ == "__main__":
    unittest.main()
