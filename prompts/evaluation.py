"""Small, deterministic end-to-end evaluation helpers for Contexto."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


EvaluationExample = Mapping[str, Any]


def _normalise(value: Any) -> str:
    return str(value).strip().casefold()


def _citation_sources(citations: Iterable[Any]) -> set[str]:
    sources = set()
    for citation in citations:
        if isinstance(citation, Mapping):
            source = citation.get("source")
        else:
            source = citation
        if source:
            sources.add(_normalise(source))
    return sources


def _context_text(chunks: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(_normalise(chunk.get("text", "")) for chunk in chunks)


def score_answer(
    example: EvaluationExample,
    answer_fn: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Score one answer for expected points, context support, and citations."""
    result = answer_fn(str(example["question"]))
    answer = _normalise(result.get("answer", ""))
    expected_points = [str(point) for point in example.get("expected_points", [])]
    retrieved_chunks = result.get("retrieved_chunks", [])
    context = _context_text(retrieved_chunks)

    point_matches = [point for point in expected_points if _normalise(point) in answer]
    supported_points = [point for point in point_matches if _normalise(point) in context]
    expected_sources = {
        _normalise(source) for source in example.get("expected_sources", set())
    }
    cited_sources = _citation_sources(result.get("sources", result.get("citations", [])))

    correctness = int(len(point_matches) == len(expected_points))
    grounding = int(len(supported_points) == len(point_matches))
    citation_accuracy = int(bool(expected_sources) and cited_sources == expected_sources)

    return {
        "question": example["question"],
        "answer": result.get("answer", ""),
        "correctness": correctness,
        "grounding": grounding,
        "citation_accuracy": citation_accuracy,
        "citations": result.get("sources", result.get("citations", [])),
    }


def evaluate_test_set(
    test_set: Sequence[EvaluationExample],
    answer_fn: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate all examples and summarize failures by dimension."""
    rows = [score_answer(example, answer_fn) for example in test_set]
    questions = len(rows)

    if not rows:
        return {
            "questions": 0,
            "avg_correctness": 0.0,
            "avg_grounding": 0.0,
            "avg_citation_accuracy": 0.0,
            "failures": [],
        }

    summary = {
        "questions": questions,
        "avg_correctness": sum(row["correctness"] for row in rows) / questions,
        "avg_grounding": sum(row["grounding"] for row in rows) / questions,
        "avg_citation_accuracy": sum(row["citation_accuracy"] for row in rows) / questions,
        "failures": [
            row
            for row in rows
            if min(row["correctness"], row["grounding"], row["citation_accuracy"]) < 1
        ],
    }
    return summary