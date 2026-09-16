import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vector_store import build_record, create_collection, create_client, readback_record, upsert_record


class VectorStoreTests(unittest.TestCase):
    def test_build_record_validates_dimension(self):
        record = build_record(
            "account-guide.md:0",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Password reset instructions for learner accounts.",
            {"source": "account-guide.md", "chunk_index": 0, "section": "Account access"},
            vector_dimension=6,
        )

        self.assertEqual(record["id"], "account-guide.md:0")
        self.assertEqual(len(record["vector"]), 6)
        self.assertEqual(record["metadata"]["source"], "account-guide.md")

    def test_insert_and_read_back_record_in_memory_collection(self):
        client = create_client(path=str(Path(__file__).resolve().parents[1] / "tmp_vector_store"))
        collection = create_collection(client, name="test-rag-chunks", vector_dimension=6, metric="cosine")
        record = build_record(
            "account-guide.md:0",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Password reset instructions for learner accounts.",
            {"source": "account-guide.md", "chunk_index": 0, "section": "Account access"},
            vector_dimension=6,
        )

        stored = upsert_record(collection, record)
        readback = readback_record(collection, "account-guide.md:0")

        self.assertEqual(stored["id"], record["id"])
        self.assertEqual(len(readback["vector"]), 6)
        self.assertEqual(readback["text"], record["text"])
        self.assertEqual(readback["metadata"], record["metadata"])


if __name__ == "__main__":
    unittest.main()
