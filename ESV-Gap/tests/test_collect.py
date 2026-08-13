import unittest
from unittest.mock import Mock, patch

import requests

from src.collect import CollectionAPIError, search_papers


class SearchPapersRetryTests(unittest.TestCase):
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
