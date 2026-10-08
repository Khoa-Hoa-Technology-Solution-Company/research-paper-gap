"""Check provenance/budget invariants and independently known score cases."""
import unittest

from src.context_retrieval import retrieve_context, windows, word_overlap


class ContextRetrievalTests(unittest.TestCase):
    def test_exact_identity_scores_one(self):
        metrics = word_overlap("alpha beta gamma", "alpha beta gamma")
        for name in ["rouge1_f1", "rouge2_f1", "rougel_f1"]:
            self.assertAlmostEqual(metrics[name], 1.)

    def test_lcs_distinguishes_bag_overlap(self):
        metrics = word_overlap("a b c", "c b a")
        self.assertEqual(metrics["rouge1_f1"], 1.)
        self.assertEqual(metrics["rouge2_f1"], 0.)
        self.assertAlmostEqual(metrics["rougel_f1"], 1/3)

    def test_no_overlap_or_empty_scores_zero(self):
        self.assertEqual(word_overlap("x y", "a b")["rougel_f1"], 0.)
        self.assertEqual(word_overlap("", "a b")["rouge1_f1"], 0.)

    def test_source_offsets_and_budget_across_methods(self):
        source = "\n".join(" ".join(f"term{j}" for j in range(i,i+100))
                           for i in [0,100,200,300])
        for method in ["bm25", "tfidf", "prior_only", "bm25_diverse", "context", "context_diverse"]:
            rows = retrieve_context(source, "term200 term250", prior={"term200": 2.},
                                    method=method, budget=117)
            self.assertEqual(sum(r["word_count"] for r in rows), 117)
            for row in rows:
                self.assertEqual(source[row["start"]:row["end"]], row["text"])
                self.assertIs(row["reviewed"], False)
            for a,b in zip(rows,rows[1:]):
                self.assertLessEqual(a["end"], b["start"])

    def test_no_text_produces_no_quotes(self):
        self.assertEqual(windows(""), [])
        self.assertEqual(retrieve_context("", "query"), [])

    def test_nonpositive_budget_rejected(self):
        with self.assertRaises(ValueError):
            retrieve_context("text", "query", budget=0)


if __name__ == "__main__":
    unittest.main()
