"""Retrieval-quality checks used to prevent unsupported answers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


MIN_TOP_SCORE = 0.72
MIN_SUPPORTING_CHUNKS = 1
REFUSAL_MESSAGE = "I don't have enough reliable context to answer that."


def retrieval_is_strong(
    chunks: Sequence[dict[str, Any]],
    *,
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> bool:
    """Return whether retrieval produced enough high-scoring evidence."""
    if not chunks:
        return False
    if min_supporting_chunks <= 0:
        raise ValueError("min_supporting_chunks must be greater than 0")

    strong_chunks = [
        chunk for chunk in chunks
        if float(chunk.get("score", 0.0)) >= min_top_score
    ]
    return len(strong_chunks) >= min_supporting_chunks


def guarded_answer(
    question: str,
    retrieve_fn: Callable[[str], Sequence[dict[str, Any]]],
    generate_fn: Callable[[str, Sequence[dict[str, Any]]], dict[str, Any]],
    *,
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> dict[str, Any]:
    """Retrieve evidence and generate only when the evidence is strong enough."""
    chunks = list(retrieve_fn(question))

    if not retrieval_is_strong(
        chunks,
        min_top_score=min_top_score,
        min_supporting_chunks=min_supporting_chunks,
    ):
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "status": "refused_weak_context",
        }

    response = generate_fn(question, chunks)
    return {**response, "status": "answered"}