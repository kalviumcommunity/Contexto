import unittest

from src.retrieval_evaluation import evaluate, evaluate_query, LABELLED_QUERIES


class RetrievalEvaluationTests(unittest.TestCase):
    def test_labels_are_retrievable_with_larger_k(self):
        results = evaluate(10)

        self.assertEqual(results["recall"], 1.0)
        self.assertGreater(results["precision"], 0.0)

    def test_small_k_exposes_recall_failure(self):
        row = evaluate_query(LABELLED_QUERIES[2], 3)

        self.assertLess(row["recall"], 1.0)
        self.assertEqual(len(row["hits"]), 1)


if __name__ == "__main__":
    unittest.main()