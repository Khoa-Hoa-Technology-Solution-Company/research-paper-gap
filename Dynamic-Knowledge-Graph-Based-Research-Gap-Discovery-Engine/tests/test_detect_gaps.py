import unittest

import networkx as nx

from src.detect_gaps import detect_temporal_decay


def temporal_config():
    return {
        "gap_detection": {
            "temporal": {
                "decay_threshold": 0.3,
                "lookback_years": 2,
                "publication_counts": {
                    2020: 10,
                    2021: 10,
                    2022: 10,
                    2023: 10,
                    2024: 10,
                    2025: 10,
                },
                "snapshot_date": "2026-07-20",
                "exclude_incomplete_final_year": True,
            }
        },
        "gap_validation": {},
    }


class TemporalDecayTests(unittest.TestCase):
    def test_zero_to_zero_windows_are_not_reported_as_decay(self):
        graph = nx.MultiDiGraph()
        concept = "legacy database control"
        for index in range(3):
            graph.add_edge(
                concept,
                f"old evidence {index}",
                source_paper=f"old-{index}",
                year=2020,
            )
        self.assertEqual(detect_temporal_decay(graph, temporal_config()), [])

    def test_positive_baseline_followed_by_zero_is_decay(self):
        graph = nx.MultiDiGraph()
        concept = "declining database control"
        for year in (2022, 2023):
            for index in range(2):
                graph.add_edge(
                    concept,
                    f"evidence {year}-{index}",
                    source_paper=f"paper-{year}-{index}",
                    year=year,
                )
        graph.add_edge(
            "unrelated current topic",
            "current evidence",
            source_paper="current-2025",
            year=2025,
        )
        gaps = detect_temporal_decay(graph, temporal_config())
        self.assertEqual([gap["concept"] for gap in gaps], [concept])
        self.assertEqual(gaps[0]["decay_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
