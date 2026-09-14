import unittest

import numpy as np

from src.vector_similarity import cosine_similarity_matrix, tfidf_matrix


class VectorSimilarityTests(unittest.TestCase):
    def test_cosine_similarity_handles_zero_rows(self):
        result = cosine_similarity_matrix([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 0.0]])

    def test_cosine_similarity_matches_expected_geometry(self):
        result = cosine_similarity_matrix([[1.0, 0.0], [1.0, 1.0]])
        self.assertAlmostEqual(result[0, 1], 2 ** -0.5)

    def test_numpy_tfidf_ranks_related_documents_higher(self):
        matrix = tfidf_matrix([
            "IoT intrusion detection adversarial attacks",
            "robust intrusion detection for IoT attacks",
            "handwritten mathematical symbol recognition",
        ])
        similarities = cosine_similarity_matrix(matrix)
        self.assertGreater(similarities[0, 1], similarities[0, 2])


if __name__ == "__main__":
    unittest.main()
