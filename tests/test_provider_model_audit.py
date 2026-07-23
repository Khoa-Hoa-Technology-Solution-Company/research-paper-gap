import unittest

from src.provider_model_audit import build_provider_model_audit


class ProviderModelAuditTests(unittest.TestCase):
    def test_groups_calls_and_links_triples_by_extraction_call_id(self):
        calls = [
            {"purpose": "typed-triple-extraction", "call_id": "a", "status": "succeeded",
             "llm": {"model_identifier": "configured"},
             "response": {"provider_model_identifier": "provider-a"}, "attempts": [{}]},
            {"purpose": "typed-triple-extraction", "call_id": "b", "status": "succeeded",
             "llm": {"model_identifier": "configured"},
             "response": {"provider_model_identifier": "provider-b"}, "attempts": [{}, {}]},
        ]
        triples = [{"extraction_call_id": "a"}, {"extraction_call_id": "a"}, {"extraction_call_id": "b"}]
        report = build_provider_model_audit(calls, triples)
        rows = {row["provider_model_identifier"]: row for row in report["provider_model_identifier_distribution"]}
        self.assertEqual(rows["provider-a"]["triples"], 2)
        self.assertEqual(rows["provider-b"]["calls"], 1)
        self.assertEqual(report["retry_summary"]["calls_with_retry"], 1)


if __name__ == "__main__":
    unittest.main()
