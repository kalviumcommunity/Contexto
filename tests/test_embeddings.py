import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embeddings import build_records


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


if __name__ == "__main__":
    unittest.main()
