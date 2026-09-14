import unittest

from src.query_planning import (
    broadening_queries,
    fallback_search_queries,
    is_safe_plain_query,
    validate_generated_queries,
)


class QueryPlanningTests(unittest.TestCase):
    def setUp(self):
        self.topic = (
            "adversarial robustness of deep-learning intrusion detection "
            "systems for IoT networks"
        )

    def test_boolean_exact_match_queries_are_rejected(self):
        query = (
            '"adversarial robustness" AND "deep learning" AND '
            '"intrusion detection" AND "IoT networks"'
        )
        self.assertFalse(is_safe_plain_query(query, self.topic))
        self.assertEqual(
            validate_generated_queries([query] * 5, self.topic), []
        )

    def test_iot_fallback_is_plain_and_recall_oriented(self):
        queries = fallback_search_queries(self.topic)
        self.assertEqual(len(queries), 5)
        self.assertIn("deep learning IoT intrusion detection", queries)
        self.assertIn("adversarial attacks IoT IDS", queries)
        self.assertTrue(all(is_safe_plain_query(query) for query in queries))

    def test_sparse_boolean_plan_gets_five_broadening_queries(self):
        existing = [
            '"adversarial robustness" AND "deep learning" AND "IoT networks"'
        ]
        broadened = broadening_queries(self.topic, existing)
        self.assertEqual(len(broadened), 5)
        self.assertTrue(all(" AND " not in query for query in broadened))


if __name__ == "__main__":
    unittest.main()
