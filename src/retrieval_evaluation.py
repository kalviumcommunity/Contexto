"""Recall and precision evaluation for labelled retrieval queries."""

from __future__ import annotations

from pathlib import Path

from src.retrieval_tuning import retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "retrieval_evaluation_results.md"
CHUNK_SIZE = 12
K_VALUES = (3, 5, 10)


LABELLED_QUERIES = [
    {
        "query": "How can a learner reset their password?",
        "relevant_chunk_ids": {"account-guide.md:0", "account-guide.md:1"},
    },
    {
        "query": "When does the cafeteria menu change?",
        "relevant_chunk_ids": {"campus-guide.md:0", "campus-guide.md:1"},
    },
    {
        "query": "What evidence is required for project submission?",
        "relevant_chunk_ids": {"submission-rubric.md:0", "submission-rubric.md:1"},
    },
]


def chunk_id(result: dict) -> str:
    metadata = result["metadata"]
    return f"{metadata['source']}:{metadata['chunk_index']}"


def evaluate_query(item: dict, k: int) -> dict:
    results = retrieve(item["query"], k=k, chunk_size=CHUNK_SIZE)
    retrieved_ids = [chunk_id(result) for result in results]
    relevant = item["relevant_chunk_ids"]
    hits = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant]
    recall = len(hits) / len(relevant)
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
    return {
        "query": item["query"],
        "retrieved_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(relevant),
        "hits": hits,
        "recall": recall,
        "precision": precision,
    }


def evaluate(k: int) -> dict:
    rows = [evaluate_query(item, k) for item in LABELLED_QUERIES]
    return {
        "k": k,
        "rows": rows,
        "recall": sum(row["recall"] for row in rows) / len(rows),
        "precision": sum(row["precision"] for row in rows) / len(rows),
    }


def failure_cause(row: dict) -> str:
    if row["recall"] == 1.0:
        return "none"
    missing = set(row["relevant_chunk_ids"]) - set(row["retrieved_ids"])
    if any(chunk_id.endswith(":1") for chunk_id in missing):
        return "chunking or small-k limit: the second relevant chunk ranked below the cutoff"
    return "embedding or query-term mismatch: a labelled chunk was not ranked"


def write_report(evaluations: list[dict]) -> None:
    best = max(evaluations, key=lambda result: (result["recall"], result["precision"]))
    lines = [
        "# Retrieval Evaluation Results",
        "",
        f"Labels were manually defined for {len(LABELLED_QUERIES)} queries using "
        f"stable `{CHUNK_SIZE}`-word chunk IDs. Metrics are macro-averaged recall@k "
        "and precision@k.",
        "",
        "## Labelled Query Set",
        "",
        "| Query | Relevant chunk IDs |",
        "| --- | --- |",
    ]
    for item in LABELLED_QUERIES:
        labels = ", ".join(f"`{value}`" for value in sorted(item["relevant_chunk_ids"]))
        lines.append(f"| {item['query']} | {labels} |")
    lines.extend(["", "## Aggregate Metrics", "", "| k | Recall@k | Precision@k |", "| ---: | ---: | ---: |"])
    for result in evaluations:
        lines.append(f"| {result['k']} | {result['recall']:.0%} | {result['precision']:.0%} |")
    lines.extend(["", "## Query-Level Results", ""])
    for result in evaluations:
        lines.extend([f"### k={result['k']}", "", "| Query | Retrieved IDs | Hits | Recall | Precision |", "| --- | --- | --- | ---: | ---: |"])
        for row in result["rows"]:
            retrieved = ", ".join(f"`{value}`" for value in row["retrieved_ids"])
            hits = ", ".join(f"`{value}`" for value in row["hits"]) or "none"
            lines.append(f"| {row['query']} | {retrieved} | {hits} | {row['recall']:.0%} | {row['precision']:.0%} |")
        lines.append("")
    lines.extend(["## Failure Analysis", ""])
    for result in evaluations:
        for row in result["rows"]:
            if row["recall"] < 1.0:
                missing = sorted(set(row["relevant_chunk_ids"]) - set(row["retrieved_ids"]))
                lines.append(f"- **k={result['k']}, {row['query']}**: missing `{', '.join(missing)}`. Likely cause: {failure_cause(row)}.")
    lines.extend(
        [
            "",
            "## Next Improvement",
            "",
            f"Choose k={best['k']} for recall-sensitive retrieval: it reaches {best['recall']:.0%} recall@k "
            f"and {best['precision']:.0%} precision@k on this set. The k=3 failures show that the "
            "second relevant chunk is pushed below the cutoff; the next experiment should test "
            "smaller semantic chunks or re-ranking while monitoring precision and context cost.",
            "",
        ]
    )
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    results = [evaluate(k) for k in K_VALUES]
    write_report(results)
    for result in results:
        print(f"k={result['k']}: recall={result['recall']:.0%}, precision={result['precision']:.0%}")
    print(f"Report written to {OUTPUT_PATH}")