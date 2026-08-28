import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunker import fixed_chunks, paragraph_chunks


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
                {"source", "chunk_index", "char_start", "char_end", "section", "page"},
            )

    def test_paragraph_chunks_preserve_each_paragraph_position(self):
        chunks = paragraph_chunks("First paragraph.\n\nSecond paragraph.", "article.md")

        self.assertEqual([chunk["text"] for chunk in chunks], [
            "First paragraph.",
            "Second paragraph.",
        ])
        self.assertEqual(chunks[0]["metadata"]["char_start"], 0)
        self.assertEqual(chunks[1]["metadata"]["char_start"], 18)


if __name__ == "__main__":
    unittest.main()