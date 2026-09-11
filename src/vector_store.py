"""Vector database setup for Contexto embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised when dependencies are unavailable
    chromadb = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_COLLECTION_NAME = "rag_chunks"
DEFAULT_VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION") or os.getenv("EMBEDDING_DIMENSION") or "6")


def create_client(*, path: str | None = None) -> Any:
    """Create a Chroma client using memory or a persisted directory."""
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Install requirements before creating the vector store.")
    if path:
        return chromadb.PersistentClient(path)
    return chromadb.Client()


def create_collection(
    client: Any,
    *,
    name: str = DEFAULT_COLLECTION_NAME,
    vector_dimension: int | None = None,
    metric: str = "cosine",
) -> Any:
    """Create a collection with metadata that records the embedding dimension and metric."""
    dimension = vector_dimension or DEFAULT_VECTOR_DIMENSION
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": metric, "dimension": str(dimension)},
    )
    return collection


def build_record(
    record_id: str,
    vector: list[float] | tuple[float, ...],
    text: str,
    metadata: dict[str, Any],
    *,
    vector_dimension: int | None = None,
) -> dict[str, Any]:
    """Validate and build a record that stores vector, text, and metadata together."""
    dimension = vector_dimension or DEFAULT_VECTOR_DIMENSION
    vector_list = [float(value) for value in vector]
    if len(vector_list) != dimension:
        raise ValueError(f"Expected vector dimension {dimension}, got {len(vector_list)}.")
    return {
        "id": record_id,
        "vector": vector_list,
        "text": text,
        "metadata": metadata,
    }


def upsert_record(collection: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Insert a single record and return the readback payload for validation."""
    collection.add(
        ids=[record["id"]],
        embeddings=[record["vector"]],
        documents=[record["text"]],
        metadatas=[record["metadata"]],
    )
    return readback_record(collection, record["id"])


def readback_record(collection: Any, record_id: str) -> dict[str, Any]:
    """Fetch and validate the stored vector, text, and metadata for a single record."""
    result = collection.get(ids=[record_id], include=["embeddings", "documents", "metadatas"])
    if not result["ids"]:
        raise KeyError(f"No stored record found for id {record_id!r}.")

    item = {
        "id": result["ids"][0],
        "vector": list(result["embeddings"][0]),
        "text": result["documents"][0],
        "metadata": result["metadatas"][0],
    }
    return item


def render_vector_store_report(
    collection_name: str,
    vector_dimension: int,
    record: dict[str, Any],
    stored_record: dict[str, Any],
) -> str:
    """Render a small markdown report showing the collection setup and readback result."""
    lines = [
        "# Vector Store Setup Report",
        "",
        f"- Collection: {collection_name}",
        f"- Vector dimension: {vector_dimension}",
        f"- Metric: cosine",
        "",
        "## Stored record",
        "",
        "```json",
        json.dumps(
            {
                "id": stored_record["id"],
                "vector_length": len(stored_record["vector"]),
                "text": stored_record["text"],
                "metadata": stored_record["metadata"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Validation",
        "",
        f"- ID matches: {stored_record['id'] == record['id']}",
        f"- Vector length matches: {len(stored_record['vector']) == len(record['vector'])}",
        f"- Text matches: {stored_record['text'] == record['text']}",
        f"- Metadata matches: {stored_record['metadata'] == record['metadata']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up a Chroma collection and read back a test record.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Collection name for the vector store.")
    parser.add_argument("--dimension", type=int, default=DEFAULT_VECTOR_DIMENSION, help="Embedding dimension for the collection.")
    parser.add_argument("--path", default=None, help="Optional directory for a persisted Chroma DB.")
    args = parser.parse_args()

    client = create_client(path=args.path)
    collection = create_collection(client, name=args.collection, vector_dimension=args.dimension, metric="cosine")
    record = build_record(
        "account-guide.md:0",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][: args.dimension],
        "Password reset instructions for learner accounts.",
        {
            "source": "account-guide.md",
            "chunk_index": 0,
            "section": "Account access",
        },
        vector_dimension=args.dimension,
    )
    stored_record = upsert_record(collection, record)

    print("collection:", args.collection)
    print("vector dimension:", args.dimension)
    print("readback id:", stored_record["id"])
    print("vector length:", len(stored_record["vector"]))
    print("text:", stored_record["text"])
    print("metadata:", stored_record["metadata"])

    output_path = PROJECT_ROOT / "outputs" / "vector_store_results.md"
    output_path.write_text(render_vector_store_report(args.collection, args.dimension, record, stored_record), encoding="utf-8")
    print(f"Saved vector store report: {output_path}")


if __name__ == "__main__":
    main()
