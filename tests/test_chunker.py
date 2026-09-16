import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunker import fixed_chunks, paragraph_chunks, token_chunks


class ChunkMetadataTests(unittest.TestCase):
    def test_fixed_chunks_include_consistent_source_metadata(self):
        chunks = fixed_chunks("alpha beta gamma", "notes.txt", size=10, overlap=2)

        self.assertTrue(chunks)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["metadata"]["source"], "notes.txt")
            self.assertEqual(chunk["metadata"]["chunk_index"], index)
            self.assertEqual(
                chunk["text"], "alpha beta gamma"[
                    chunk["metadata"]["char_start"]:chunk["metadata"]["char_end"]
                ].strip()
            )
            self.assertEqual(
                set(chunk["metadata"]),
                {
                    "source", "chunk_index", "char_start", "char_end", "section", "page",
                    "token_start", "token_end", "token_count",
                },
            )

    def test_paragraph_chunks_preserve_each_paragraph_position(self):
        chunks = paragraph_chunks("First paragraph.\n\nSecond paragraph.", "article.md")

        self.assertEqual([chunk["text"] for chunk in chunks], [
            "First paragraph.",
            "Second paragraph.",
        ])
        self.assertEqual(chunks[0]["metadata"]["char_start"], 0)
        self.assertEqual(chunks[1]["metadata"]["char_start"], 18)

    def test_token_chunks_limit_size_and_repeat_overlap(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = token_chunks(text, "demo.txt", size=4, overlap=2)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            metadata = chunk["metadata"]
            self.assertLessEqual(metadata["token_count"], 4)
            self.assertEqual(metadata["source"], "demo.txt")

        first = chunks[0]["metadata"]
        second = chunks[1]["metadata"]
        self.assertEqual(second["token_start"], first["token_end"] - 2)
        self.assertEqual(second["token_end"] - second["token_start"], 4)
        self.assertIn("beta", chunks[0]["text"])
        self.assertIn("beta", chunks[1]["text"])

    def test_token_chunks_reject_invalid_overlap(self):
        with self.assertRaises(ValueError):
            token_chunks("text", "demo.txt", size=4, overlap=4)


if __name__ == "__main__":
    unittest.main()