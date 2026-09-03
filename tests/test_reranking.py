import unittest

from src.reranking import CANDIDATE_K, FINAL_K, QUERY, rerank_score, run_reranking


class RerankingTests(unittest.TestCase):
    def test_candidate_set_is_larger_than_final_context(self):
        candidates, final_chunks = run_reranking()

        self.assertLessEqual(len(candidates), CANDIDATE_K)
        self.assertEqual(len(final_chunks), FINAL_K)
        self.assertGreater(len(candidates), len(final_chunks))

    def test_reranking_selects_submission_evidence(self):
        _, final_chunks = run_reranking()

        self.assertEqual(final_chunks[0]["metadata"]["source"], "submission-rubric.md")
        self.assertGreater(rerank_score(QUERY, final_chunks[0]), final_chunks[-1]["rerank_score"])


if __name__ == "__main__":
    unittest.main()