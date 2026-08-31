"""Run the full ingestion pipeline and validate every source file is accounted for."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.token_chunker import token_chunks

ENCODING_NAME = "cl100k_base"


def load_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def clean(text: str) -> str:
    """Normalize whitespace so chunk boundaries are stable and readable."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def tag_chunks(source_name: str, text_chunks: list[str]) -> list[dict[str, object]]:
    """Attach metadata to each chunk so source and boundary details remain visible."""
    encoder = tiktoken.get_encoding(ENCODING_NAME)
    tagged: list[dict[str, object]] = []
    for chunk_index, chunk in enumerate(text_chunks, start=1):
        tagged.append(
            {
                "source": source_name,
                "chunk_index": chunk_index,
                "text": chunk,
                "metadata": {
                    "source_file": source_name,
                    "chunk_index": chunk_index,
                    "token_count": len(encoder.encode(chunk)),
                },
            }
        )
    return tagged


def ingest(folder: str | Path) -> tuple[list[Path], int, list[dict[str, object]], list[tuple[str, str]]]:
    """Load, clean, chunk, and tag every file in a directory tree."""
    root = Path(folder)
    files = [path for path in root.rglob("*") if path.is_file()]
    docs = 0
    chunks: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for path in files:
        try:
            raw_text = load_text(path)
            cleaned = clean(raw_text)
            if not cleaned:
                raise ValueError("empty document after cleaning")
            chunk_list = token_chunks(cleaned, size=200, overlap=30)
            chunks.extend(tag_chunks(path.name, chunk_list))
            docs += 1
        except Exception as exc:  # pragma: no cover - demo code intentionally reports failures
            failures.append((path.name, str(exc)))

    return files, docs, chunks, failures


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    files, docs, chunks, failures = ingest(data_dir)

    print(f"files={len(files)} docs={docs} chunks={len(chunks)} failures={len(failures)}")
    assert docs + len(failures) == len(files), "a document was silently dropped!"

    for name, err in failures:
        print(f"FAILED: {name} {err}")

    if chunks:
        sample = chunks[0]
        print("sample:", sample["text"][:80], "|", sample["metadata"])
    else:
        print("No files found in data/. Add source documents to test the ingest pipeline.")


if __name__ == "__main__":
    main()
