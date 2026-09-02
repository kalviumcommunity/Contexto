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


def _normalize_search_result(result: Any) -> dict[str, Any]:
    """Normalize a search result from Chroma into a consistent format.
    
    Handles both dict-based and tuple-based result formats from different
    versions and configurations of Chroma.
    """
    if isinstance(result, dict):
        return result
    
    if isinstance(result, (list, tuple)) and len(result) >= 3:
        # Handle tuple-based format: (id, distance, metadata, document)
        result_id = result[0] if len(result) > 0 else ""
        distance = result[1] if len(result) > 1 else 0.0
        metadata = result[2] if len(result) > 2 else {}
        document = result[3] if len(result) > 3 else ""
        
        return {
            "id": result_id,
            "score": float(distance),
            "metadata": metadata if isinstance(metadata, dict) else {},
            "text": document if isinstance(document, str) else "",
        }
    
    return {"id": "", "score": 0.0, "metadata": {}, "text": ""}


def retrieve(
    collection: Any,
    query_vector: list[float],
    *,
    k: int = 3,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search the vector database for the k most similar chunks to a query vector.
    
    Args:
        collection: The Chroma collection to search.
        query_vector: The embedding vector of the user's query.
        k: Number of top results to return. Default is 3.
        metadata_filter: Optional dict to filter results by metadata.
                        E.g., {"section": "Account access"} or
                             {"source": "article.txt", "chunk_index": {"$gte": 0}}
    
    Returns:
        A list of results, each containing:
            - score: Similarity score (higher is more similar)
            - text: The chunk text
            - metadata: Source, chunk_index, and section information
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=metadata_filter,
        )
    except Exception:
        # Fallback for other collection APIs
        results = collection.search(
            query_vector=query_vector,
            top_k=k,
            filter=metadata_filter,
        )
    
    # Normalize Chroma's response format
    retrieved = []
    
    if isinstance(results, dict) and "ids" in results:
        # Standard Chroma dict response format
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for idx, (result_id, distance, document, metadata) in enumerate(
            zip(ids, distances, documents, metadatas)
        ):
            retrieved.append({
                "score": float(distance),
                "text": document or "",
                "metadata": metadata or {},
            })
    elif isinstance(results, list):
        # List-based or iterator response format
        for result in results:
            retrieved.append(_normalize_search_result(result))
    
    return retrieved


def retrieve_with_embedding(
    collection: Any,
    query: str,
    embed_fn: Any,
    *,
    k: int = 3,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and retrieve the top-k similar chunks.
    
    This is the high-level retrieval function that combines embedding
    and search into one step.
    
    Args:
        collection: The Chroma collection to search.
        query: The user's query string.
        embed_fn: A callable that takes a list of texts and returns embeddings.
                 E.g., lambda texts: embed_texts(client, texts)
        k: Number of top results to return. Default is 3.
        metadata_filter: Optional dict to filter results by metadata.
    
    Returns:
        A list of results with score, text, and metadata.
    """
    query_embedding = embed_fn([query])[0]
    return retrieve(collection, query_embedding, k=k, metadata_filter=metadata_filter)


def keyword_score(text: str, keywords: Sequence[str]) -> float:
    """Calculate keyword match score as sum of keyword occurrences in text.
    
    Case-insensitive matching. Each keyword occurrence increases the score by 1.
    Useful for measuring lexical similarity (exact terms, product names, IDs).
    
    Args:
        text: The text to search in.
        keywords: List of keywords to find.
    
    Returns:
        Number of keyword matches found (can be > 1 if keywords repeat).
    """
    if not keywords:
        return 0.0
    
    lowered = text.lower()
    score = 0
    for keyword in keywords:
        if keyword:  # Skip empty keywords
            score += lowered.count(keyword.lower())
    return float(score)


def hybrid_rank(
    vector_results: Sequence[dict[str, Any]],
    keywords: Sequence[str],
    *,
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2,
) -> list[dict[str, Any]]:
    """Combine vector similarity and keyword matching scores for hybrid ranking.
    
    Performs a weighted combination of:
    - Vector score (semantic similarity)
    - Keyword score (lexical/exact match)
    
    This approach helps when users search for exact product names, IDs, 
    codes, or specific terms where pure semantic similarity might miss matches.
    
    Args:
        vector_results: Results from vector search with 'score' and 'text' keys.
        keywords: List of keywords to match.
        vector_weight: Weight for vector similarity score (default 0.8).
        keyword_weight: Weight for keyword matching (default 0.2).
    
    Returns:
        Results ranked by hybrid score, with new 'keyword_score' and 
        'hybrid_score' fields added.
    
    Raises:
        ValueError: If weights don't sum to approximately 1.0.
    """
    total_weight = vector_weight + keyword_weight
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"weights must sum to ~1.0, got {total_weight}")
    
    ranked = []
    for item in vector_results:
        # Calculate keyword score (number of keyword matches)
        lexical_score = keyword_score(item["text"], keywords)
        
        # Normalize lexical score to 0-1 range if needed
        # Use min() to cap at 1.0 for comparison with normalized vector score
        normalized_lexical = min(lexical_score / max(len(keywords), 1), 1.0)
        
        # Combine scores
        combined_score = (
            (vector_weight * item["score"]) +
            (keyword_weight * normalized_lexical)
        )
        
        ranked.append({
            **item,
            "keyword_score": lexical_score,
            "hybrid_score": combined_score,
        })
    
    # Sort by hybrid score descending
    return sorted(ranked, key=lambda item: item["hybrid_score"], reverse=True)


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
