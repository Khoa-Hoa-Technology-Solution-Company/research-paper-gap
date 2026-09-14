import unittest
from unittest.mock import patch

from src.closure_search import (
    candidate_search_queries,
    candidate_search_query,
    search_candidate,
    search_openalex,
    search_semantic_scholar,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class ClosureSearchTests(unittest.TestCase):
    def test_long_extracted_clauses_are_compacted_for_api_search(self):
        queries = candidate_search_queries({
            "type": "evidence_gap",
            "subject": "A very long paper title about monolithic software systems and deployment",
            "missing_capability": (
                "a high-performance design for encryption in LSM-KVS, often "
                "focused on in-memory protection with excessive overhead"
            ),
        }, "security of monolith")
        self.assertTrue(queries)
        self.assertTrue(all(len(query.split()) <= 25 for query in queries))
        self.assertTrue(all("often" not in query for query in queries))

    def test_evidence_gap_query_contains_both_scoped_concepts(self):
        query = candidate_search_query({
            "type": "evidence_gap",
            "subject": "visual encoder",
            "missing_capability": "handwritten expression robustness",
        })
        self.assertEqual(query, "visual encoder handwritten expression robustness")

    def test_closure_uses_complementary_query_formulations(self):
        queries = candidate_search_queries({
            "type": "evidence_gap",
            "subject": "visual encoder",
            "missing_capability": "handwritten expression robustness",
        }, domain="mathematical expression recognition")
        self.assertEqual(len(queries), 3)
        self.assertEqual(len(set(queries)), 3)
        self.assertIn("mathematical expression recognition", queries[1])

    def test_search_returns_bounded_paper_metadata(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(payload={"data": [
                {"paperId": "p1", "title": "Closing work"},
                {"paperId": "p2", "title": "Other work"},
            ]})

        papers = search_semantic_scholar(
            "visual encoder robustness",
            [2020, 2026],
            max_results=1,
            request_get=fake_get,
        )
        self.assertEqual([paper["paperId"] for paper in papers], ["p1"])
        self.assertEqual(calls[0][1]["params"]["year"], "2020-2026")

    def test_openalex_results_are_normalised_for_closure_audit(self):
        def fake_get(_url, **_kwargs):
            return FakeResponse(payload={"results": [{
                "id": "https://openalex.org/W1",
                "title": "Closing work",
                "publication_year": 2025,
                "abstract_inverted_index": {
                    "robust": [0],
                    "recognition": [1],
                },
                "cited_by_count": 3,
            }]})

        papers = search_openalex(
            "robust recognition",
            [2020, 2026],
            request_get=fake_get,
        )
        self.assertEqual(papers[0]["paperId"], "https://openalex.org/W1")
        self.assertEqual(papers[0]["abstract"], "robust recognition")
        self.assertEqual(papers[0]["year"], 2025)

    @patch("src.closure_search.time.sleep")
    @patch("src.closure_search.search_semantic_scholar")
    @patch("src.closure_search.search_openalex", return_value=[])
    def test_no_semantic_scholar_key_prefers_openalex(
        self,
        mock_openalex,
        mock_semantic_scholar,
        _mock_sleep,
    ):
        record = search_candidate({
            "type": "evidence_gap",
            "subject": "visual encoder",
            "missing_capability": "handwriting robustness",
        }, {
            "project": {"domain": "expression recognition"},
            "collection": {"year_range": [2020, 2026]},
            "api_keys": {"semantic_scholar": "", "openalex": ""},
            "gap_validation": {"external_closure": {
                "enabled": True,
                "queries_per_candidate": 3,
                "prefer_openalex_without_semantic_scholar_key": True,
                "delay_between_queries_seconds": 0,
            }},
        })
        self.assertTrue(record["performed"])
        self.assertEqual(record["completed_query_count"], 3)
        self.assertEqual(mock_openalex.call_count, 3)
        mock_semantic_scholar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
