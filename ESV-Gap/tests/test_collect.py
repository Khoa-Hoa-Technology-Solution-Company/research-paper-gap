import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from src.collect import CollectionAPIError, collect_papers, search_papers


class SearchPapersRetryTests(unittest.TestCase):
    @patch("src.collect.time.sleep")
    @patch("src.collect.search_papers")
    def test_sparse_boolean_plan_is_adaptively_broadened(
        self, mock_search, _mock_sleep
    ):
        boolean_queries = [f'"strict {index}" AND "IoT"' for index in range(5)]

        def fake_search(query, **_kwargs):
            if query in boolean_queries:
                index = boolean_queries.index(query)
                return [{
                    "paperId": f"strict-{index}", "title": f"Strict {index}",
                    "abstract": "relevant " * 120, "year": 2025,
                    "citationCount": 1,
                }]
            prefix = query.replace(" ", "-")[:20]
            return [{
                "paperId": f"{prefix}-{index}", "title": f"Broad {index}",
                "abstract": "relevant " * 120, "year": 2024,
                "citationCount": 0,
            } for index in range(10)]

        mock_search.side_effect = fake_search
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = {
                "project": {"domain": (
                    "adversarial robustness of deep-learning intrusion detection "
                    "systems for IoT networks"
                )},
                "api_keys": {"semantic_scholar": ""},
                "collection": {
                    "queries": boolean_queries,
                    "year_range": [2019, 2026], "max_papers": 100,
                    "delay_between_requests": 0, "max_retries": 1,
                    "request_timeout_seconds": 1, "max_retry_wait_seconds": 0,
                    "adaptive_query_broadening": True,
                    "min_raw_papers_before_broadening": 30,
                },
                "filtering": {
                    "min_abstract_length": 1, "target_corpus_size": 50,
                },
                "paths": {"raw_data": str(root / "raw")},
            }
            papers = collect_papers(config)

        self.assertGreaterEqual(len(papers), 30)
        searched_queries = [
            call.kwargs.get("query", call.args[0] if call.args else "")
            for call in mock_search.call_args_list
        ]
        self.assertIn("deep learning IoT intrusion detection", searched_queries)
        self.assertGreater(len(searched_queries), len(boolean_queries))

    @patch("src.collect.time.sleep")
    @patch("src.collect.requests.get")
    def test_transient_network_error_recovers(self, mock_get, mock_sleep):
        success = Mock(status_code=200)
        success.raise_for_status.return_value = None
        success.json.return_value = {
            "data": [{"paperId": "paper-1", "title": "Recovered"}],
            "total": 1,
        }
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("temporary failure"),
            success,
        ]

        papers = search_papers(
            "research gaps",
            [2020, 2025],
            delay=1,
            max_results=10,
            max_retries=3,
        )

        self.assertEqual([paper["paperId"] for paper in papers], ["paper-1"])
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch("src.collect.time.sleep")
    @patch("src.collect.requests.get")
    def test_permanent_network_error_fails_after_bounded_retries(
        self, mock_get, mock_sleep
    ):
        mock_get.side_effect = requests.exceptions.ConnectionError("blocked")

        with self.assertRaisesRegex(CollectionAPIError, "after 3 attempts"):
            search_papers(
                "research gaps",
                [2020, 2025],
                delay=1,
                max_retries=3,
            )

        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("src.collect.time.sleep")
    @patch("src.collect.requests.get")
    def test_rate_limit_is_also_bounded(self, mock_get, mock_sleep):
        rate_limited = Mock(status_code=429, headers={"Retry-After": "120"})
        mock_get.return_value = rate_limited

        with self.assertRaisesRegex(CollectionAPIError, "rate limit persisted"):
            search_papers(
                "research gaps",
                [2020, 2025],
                max_retries=2,
                max_retry_wait=7,
            )

        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
