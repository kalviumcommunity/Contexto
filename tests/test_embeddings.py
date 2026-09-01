import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embeddings import (
    batches,
    build_records,
    cosine_similarity,
    estimate_tokens,
    rank_chunks_for_query,
    run_batch_embedding,
)


class EmbeddingRecordTests(unittest.TestCase):
    def test_build_records_keep_text_and_metadata_together_with_vectors(self):
        chunks = [
            {
                "text": "Password reset instructions for learner accounts.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Overview", "page": 1},
            },
            {
                "text": "Learners can recover access using their registered email.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Overview", "page": 1},
            },
        ]

        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        records = build_records(chunks, vectors)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], chunks[0]["text"])
        self.assertEqual(records[0]["metadata"]["source"], "account-guide.md")
        self.assertEqual(records[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(len(records[0]["embedding"]), 3)
        self.assertEqual(records[1]["embedding"][-1], 0.6)

    def test_cosine_similarity_reaches_one_for_same_vector_and_zero_for_orthogonal_vector(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_rank_chunks_for_query_places_password_recovery_chunks_first(self):
        query = [1.0, 0.9, 0.7]
        records = [
            {
                "text": "The cafeteria menu changes every Friday.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 3},
                "embedding": [0.2, 0.1, 0.0],
            },
            {
                "text": "Password reset instructions for learner accounts.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0},
                "embedding": [0.9, 0.8, 0.7],
            },
            {
                "text": "Learners can recover access using their registered email.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1},
                "embedding": [0.8, 0.7, 0.6],
            },
        ]

        ranked = rank_chunks_for_query(query, records)

        relevant_texts = {
            "Password reset instructions for learner accounts.",
            "Learners can recover access using their registered email.",
        }
        self.assertIn(ranked[0]["text"], relevant_texts)
        self.assertIn(ranked[1]["text"], relevant_texts)
        self.assertEqual(ranked[-1]["text"], "The cafeteria menu changes every Friday.")
        self.assertGreater(ranked[0]["score"], ranked[-1]["score"])

    def test_batches_split_items_by_requested_size(self):
        grouped = list(batches([1, 2, 3, 4, 5], 2))
        self.assertEqual(grouped, [[1, 2], [3, 4], [5]])

    def test_run_batch_embedding_skips_existing_chunks_and_reports_cost(self):
        class FakeEmbeddingResponse:
            def __init__(self, vectors):
                self.data = [type("Item", (), {"embedding": vector})() for vector in vectors]

        class FakeClient:
            def __init__(self):
                self.calls = []

            @property
            def embeddings(self):
                return self

            def create(self, model, input):
                self.calls.append(list(input))
                return FakeEmbeddingResponse([[0.1, 0.2, 0.3] for _ in input])

        chunks = [
            {"id": "chunk-1", "text": "one two three four", "metadata": {"source": "a.md"}},
            {"id": "chunk-2", "text": "reset password access", "metadata": {"source": "b.md"}},
            {"id": "chunk-3", "text": "support agent verify identity", "metadata": {"source": "c.md"}},
        ]

        client = FakeClient()
        records, summary = run_batch_embedding(
            chunks,
            client=client,
            model="demo-model",
            batch_size=2,
            existing_ids={"chunk-2"},
            price_per_1k_tokens=0.00002,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["skipped_existing"], 1)
        self.assertEqual(summary["embedded"], 2)
        self.assertEqual(summary["total_chunks"], 3)
        self.assertGreater(summary["estimated_cost_usd"], 0.0)
        self.assertEqual(summary["attempted_batches"], 1)
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
