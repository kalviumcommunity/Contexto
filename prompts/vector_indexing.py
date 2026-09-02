"""Utilities for preparing chunk embeddings for storage in a vector database."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any


def to_vector_record(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert a chunk dict into a vector-db record with text and metadata.

    The returned structure mirrors the assignment contract used for indexing and
    later retrieval: each record is keyed by a stable chunk id and contains the
    embedding, the original chunk text, and only the metadata needed for filters.
    """
    metadata = chunk["metadata"]
    return {
        "id": chunk["id"],
        "vector": chunk["embedding"],
        "text": chunk["text"],
        "metadata": {
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "section": metadata.get("section"),
        },
    }


def batches(items: Sequence[Any] | Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield a sequence in fixed-size batches."""
    if size <= 0:
        raise ValueError("size must be greater than 0")

    item_list = list(items)
    for start in range(0, len(item_list), size):
        yield item_list[start : start + size]


def _upsert_collection(collection: Any, records: list[dict[str, Any]]) -> None:
    """Upsert records using either a dict-based or Chroma-style API."""
    if not records:
        return

    try:
        collection.upsert(records)
        return
    except TypeError:
        pass

    ids = [record["id"] for record in records]
    embeddings = [record["vector"] for record in records]
    documents = [record["text"] for record in records]
    metadatas = [record["metadata"] for record in records]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def index_chunks(collection: Any, embedded_chunks: Sequence[dict[str, Any]], *, batch_size: int = 100) -> dict[str, Any]:
    """Convert chunks into vector records, upsert them in batches, and validate the count."""
    records = [to_vector_record(chunk) for chunk in embedded_chunks]
    inserted = 0
    failures: list[dict[str, Any]] = []

    for batch in batches(records, batch_size):
        try:
            _upsert_collection(collection, batch)
            inserted += len(batch)
        except Exception as error:  # pragma: no cover - demo/developer debugging path
            failures.append({"batch_start_id": batch[0]["id"], "error": str(error)})

    indexed_count = collection.count() if hasattr(collection, "count") else len(records)
    expected_count = len(embedded_chunks)

    result = {
        "expected_count": expected_count,
        "inserted_this_run": inserted,
        "indexed_count": indexed_count,
        "failures": failures,
        "records": records,
    }

    print("expected chunks:", expected_count)
    print("inserted this run:", inserted)
    print("indexed count:", indexed_count)
    print("failures:", failures)
    assert indexed_count == expected_count, "indexed count does not match chunk count"
    return result


def get_stored_record(collection: Any, record_id: str) -> dict[str, Any]:
    """Normalize a record fetched from a vector database into the shape used by the app."""
    stored = collection.get(record_id)
    if not isinstance(stored, dict):
        raise TypeError("collection.get() did not return a mapping")

    if "text" in stored and "vector" in stored:
        return stored

    ids = stored.get("ids", [])
    documents = stored.get("documents", [])
    embeddings = stored.get("embeddings", []) or stored.get("vectors", [])
    metadatas = stored.get("metadatas", [])

    if not ids:
        raise KeyError(f"record {record_id!r} not found")

    idx = ids.index(record_id) if record_id in ids else 0
    return {
        "id": record_id,
        "vector": embeddings[idx] if idx < len(embeddings) else [],
        "text": documents[idx] if idx < len(documents) else "",
        "metadata": metadatas[idx] if idx < len(metadatas) else {},
    }


def spot_check_index(collection: Any, embedded_chunks: Sequence[dict[str, Any]], sample_index: int = 0) -> dict[str, Any]:
    """Read back one embedded chunk and assert it matches the source data."""
    sample = embedded_chunks[sample_index]
    stored = get_stored_record(collection, sample["id"])

    assert stored["text"] == sample["text"]
    assert stored["metadata"]["source"] == sample["metadata"]["source"]
    assert len(stored["vector"]) == len(sample["embedding"])

    print("spot check passed:", sample["id"])
    print("source:", stored["metadata"]["source"])
    print("text preview:", stored["text"][:120])
    return {"sample_id": sample["id"], "stored": stored}


def main() -> None:
    """Simple demo showing how to prepare, index, and verify stored chunk records."""
    try:
        import chromadb
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("chromadb is required to run the vector index demo") from exc

    client = chromadb.PersistentClient(path="./.chroma")
    collection = client.get_or_create_collection("contexto_chunks")

    sample_chunks = [
        {
            "id": "doc-1:0",
            "embedding": [0.1, 0.2, 0.3],
            "text": "The newsroom verifies every claim against the original source notes.",
            "metadata": {
                "source": "sample_article_1.txt",
                "chunk_index": 1,
                "section": "Background",
            },
        },
        {
            "id": "doc-1:1",
            "embedding": [0.2, 0.3, 0.4],
            "text": "A fact is only trustworthy when the source, date, and quote are preserved.",
            "metadata": {
                "source": "sample_article_1.txt",
                "chunk_index": 2,
                "section": "Method",
            },
        },
    ]

    index_chunks(collection, sample_chunks, batch_size=1)
    spot_check_index(collection, sample_chunks)


if __name__ == "__main__":
    main()
