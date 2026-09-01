import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embeddings import build_records, cosine_similarity, rank_chunks_for_query


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


if __name__ == "__main__":
    unittest.main()
