"""Deterministic retrieval relevance experiment for the sample RAG corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "retrieval_tuning_results.md"


TEST_QUERIES = [
    {
        "query": "How can a learner reset their password?",
        "expected_source": "account-guide.md",
    },
    {
        "query": "When does the cafeteria menu change?",
        "expected_source": "campus-guide.md",
    },
    {
        "query": "What evidence is required for project submission?",
        "expected_source": "submission-rubric.md",
    },
]


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    doc_type: str
    chunk_index: int


DOCUMENTS = [
    (
        "account-guide.md",
        "guide",
        "Learners can reset a forgotten password from the account settings page. "
        "Choose Reset password, verify the account email, and follow the link "
        "before it expires.",
    ),
    (
        "campus-guide.md",
        "guide",
        "The cafeteria menu changes every Monday morning. The weekly menu is "
        "posted near the entrance and on the campus services page.",
    ),
    (
        "submission-rubric.md",
        "rubric",
        "Project submissions must include evidence: a working demonstration, "
        "test results, and a short explanation of design decisions.",
    ),
    (
        "news-brief.txt",
        "article",
        "The student newspaper reported on a busy campus week and interviewed "
        "several learners about their daily routines.",
    ),
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def build_chunks(chunk_size: int) -> list[Chunk]:
    """Create reproducible word chunks while retaining source metadata."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    chunks: list[Chunk] = []
    for source, doc_type, text in DOCUMENTS:
        words = text.split()
        for index in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[index:index + chunk_size])
            chunks.append(Chunk(chunk_text, source, doc_type, index // chunk_size))
    return chunks


def _score(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    return len(query_tokens & text_tokens) / len(query_tokens)


def retrieve(
    query: str,
    *,
    k: int,
    metadata_filter: dict[str, str] | None = None,
    min_score: float = 0.0,
    chunk_size: int = 40,
) -> list[dict]:
    """Rank chunks and apply metadata and score constraints."""
    if k < 1:
        raise ValueError("k must be positive")
    chunks = build_chunks(chunk_size)
    if metadata_filter:
        chunks = [
            chunk
            for chunk in chunks
            if all(getattr(chunk, key, None) == value for key, value in metadata_filter.items())
        ]

    ranked = sorted(
        (
            {
                "text": chunk.text,
                "metadata": {
                    "source": chunk.source,
                    "doc_type": chunk.doc_type,
                    "chunk_index": chunk.chunk_index,
                },
                "score": _score(query, chunk.text),
            }
            for chunk in chunks
        ),
        key=lambda result: (-result["score"], result["metadata"]["source"]),
    )
    return [result for result in ranked if result["score"] >= min_score][:k]


SETTINGS = [
    {"name": "baseline_k3", "k": 3, "filter": None, "min_score": 0.0, "chunk_size": 40},
    {"name": "filtered_k3", "k": 3, "filter": {"doc_type": "guide"}, "min_score": 0.0, "chunk_size": 40},
    {"name": "strict_k5", "k": 5, "filter": None, "min_score": 0.5, "chunk_size": 40},
]


def evaluate(setting: dict) -> list[dict]:
    rows = []
    for item in TEST_QUERIES:
        results = retrieve(
            item["query"],
            k=setting["k"],
            metadata_filter=setting["filter"],
            min_score=setting["min_score"],
            chunk_size=setting["chunk_size"],
        )
        sources = [result["metadata"]["source"] for result in results]
        rows.append(
            {
                "query": item["query"],
                "expected_source": item["expected_source"],
                "returned_sources": sources,
                "top_source": sources[0] if sources else "none",
                "hit": item["expected_source"] in sources,
            }
        )
    return rows


def run_experiment() -> list[dict]:
    summary = []
    for setting in SETTINGS:
        rows = evaluate(setting)
        hits = sum(row["hit"] for row in rows)
        top_hits = sum(row["top_source"] == row["expected_source"] for row in rows)
        summary.append(
            {
                "setting": setting,
                "hit_rate": hits / len(rows),
                "top1_hit_rate": top_hits / len(rows),
                "details": rows,
            }
        )
    return summary


def write_report(summary: list[dict]) -> None:
    best = max(summary, key=lambda row: (row["top1_hit_rate"], row["hit_rate"]))
    lines = [
        "# Retrieval Tuning Results",
        "",
        "This deterministic experiment evaluates three queries against a small "
        "metadata-aware corpus. Relevance is measured by source hit rate and top-1 hit rate.",
        "",
        "## Test Queries",
        "",
        "| Query | Expected source |",
        "| --- | --- |",
    ]
    lines.extend(f"| {item['query']} | `{item['expected_source']}` |" for item in TEST_QUERIES)
    lines.extend(["", "## Compared Settings", "", "| Setting | Chunk size | k | Filter | Min score | Hit rate | Top-1 hit rate |", "| --- | ---: | ---: | --- | ---: | ---: | ---: |"])
    for result in summary:
        setting = result["setting"]
        filter_name = setting["filter"]["doc_type"] if setting["filter"] else "none"
        lines.append(
            f"| `{setting['name']}` | {setting['chunk_size']} | {setting['k']} | {filter_name} | "
            f"{setting['min_score']:.2f} | {result['hit_rate']:.0%} | {result['top1_hit_rate']:.0%} |"
        )
    lines.extend(["", "## Query-Level Results", ""])
    for result in summary:
        lines.extend([f"### `{result['setting']['name']}`", "", "| Query | Returned sources | Hit |", "| --- | --- | :---: |"])
        for row in result["details"]:
            sources = ", ".join(f"`{source}`" for source in row["returned_sources"]) or "none"
            lines.append(f"| {row['query']} | {sources} | {'yes' if row['hit'] else 'no'} |")
        lines.append("")
    best_setting = best["setting"]
    lines.extend(
        [
            "## Decision",
            "",
            f"Choose `{best_setting['name']}`: it achieved a {best['hit_rate']:.0%} source hit rate "
            f"and {best['top1_hit_rate']:.0%} top-1 hit rate on all three queries. Its guide filter "
            "was not used, which avoids dropping the rubric source needed by the third query. "
            "This is a small offline benchmark, so it should be rerun with production queries before rollout.",
            "",
        ]
    )
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    results = run_experiment()
    write_report(results)
    for result in results:
        print(f"{result['setting']['name']}: hit_rate={result['hit_rate']:.0%}, top1={result['top1_hit_rate']:.0%}")
    print(f"Report written to {OUTPUT_PATH}")