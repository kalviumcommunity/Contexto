import unittest

from src.retrieval_tuning import evaluate, run_experiment


class RetrievalTuningTests(unittest.TestCase):
    def test_baseline_retrieves_expected_source_for_every_query(self):
        rows = evaluate({
            "name": "baseline_k3",
            "k": 3,
            "filter": None,
            "min_score": 0.0,
            "chunk_size": 40,
        })

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["hit"] for row in rows))
        self.assertTrue(all(row["top_source"] == row["expected_source"] for row in rows))

    def test_experiment_selects_highest_scoring_setting(self):
        results = run_experiment()

        best = max(results, key=lambda row: (row["top1_hit_rate"], row["hit_rate"]))
        self.assertEqual(best["setting"]["name"], "baseline_k3")
        self.assertEqual(best["hit_rate"], 1.0)
        self.assertEqual(best["top1_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()