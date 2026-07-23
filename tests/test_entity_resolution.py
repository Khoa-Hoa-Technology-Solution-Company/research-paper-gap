import unittest
from unittest.mock import patch

from src.entity_resolution import _resolve_entities_with_audit, resolve_entities


def triple(subject, subject_type, obj, object_type):
    return {
        "subject": subject,
        "subject_type": subject_type,
        "relation": "USES",
        "object": obj,
        "object_type": object_type,
        "confidence": 0.8,
        "evidence_quote": "Evidence.",
        "year": 2024,
    }


class EntityResolutionTests(unittest.TestCase):
    def test_lexical_merge_is_same_type_and_has_stable_canonical(self):
        triples = [
            triple("OAuth protocol", "CONCEPT", "Tool A", "TOOL"),
            triple("protocol OAuth", "CONCEPT", "Tool B", "TOOL"),
        ]
        resolved, mapping = resolve_entities(triples, enable_semantic=False)

        self.assertEqual(mapping, {"OAuth protocol": ["protocol OAuth"]})
        self.assertEqual({item["subject"] for item in resolved}, {"OAuth protocol"})

    def test_type_conflict_is_not_merged_and_is_type_qualified(self):
        triples = [
            triple("Gateway", "CONCEPT", "Policy", "CONCEPT"),
            triple("Gateway", "TOOL", "Agent", "TOOL"),
        ]
        resolved, mapping, audit = _resolve_entities_with_audit(
            triples, enable_semantic=False
        )

        self.assertEqual(mapping, {})
        self.assertEqual(
            {item["subject"] for item in resolved},
            {"Gateway [CONCEPT]", "Gateway [TOOL]"},
        )
        self.assertEqual(audit["counts"]["same-label_type-conflicts"], 1)

    def test_chaining_is_blocked_when_component_roots_fail_cohesion(self):
        triples = [
            triple("Alpha", "CONCEPT", "Tool A", "TOOL"),
            triple("B", "CONCEPT", "Tool B", "TOOL"),
            triple("Charlie", "CONCEPT", "Tool C", "TOOL"),
        ]
        scores = {
            frozenset(("alpha", "b")): 90.0,
            frozenset(("b", "charlie")): 90.0,
        }

        def fake_similarity(left, right):
            return scores.get(frozenset((left, right)), 0.0), "test-backend"

        with patch("src.entity_resolution._lexical_similarity", side_effect=fake_similarity):
            resolved, mapping, audit = _resolve_entities_with_audit(
                triples, enable_semantic=False
            )

        # "B" is the shortest representative after merging Alpha/B. A
        # representative-only guard would then merge Charlie, but complete
        # linkage checks Alpha/Charlie too and rejects the chain.
        self.assertEqual(mapping, {"B": ["Alpha"]})
        self.assertEqual({item["subject"] for item in resolved}, {"B", "Charlie"})
        self.assertEqual(audit["counts"]["lexical_cohesion_rejections"], 1)


if __name__ == "__main__":
    unittest.main()
