"""Candidate retrieval and deterministic second-stage re-ranking demo."""

from __future__ import annotations

from pathlib import Path

from src.retrieval_tuning import _tokens, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reranking_results.md"
QUERY = "What evidence is required for project submission?"
CANDIDATE_K = 10
FINAL_K = 3


def rerank_score(query: str, chunk: dict) -> float:
    """Score query/chunk pairs using coverage plus direct phrase matches."""
    query_tokens = _tokens(query)
    chunk_tokens = _tokens(chunk["text"])
    coverage = len(query_tokens & chunk_tokens) / len(query_tokens)
    direct_terms = {"evidence", "required", "project", "submission"}
    direct_coverage = len(direct_terms & chunk_tokens) / len(direct_terms)
    phrase_bonus = 0.25 if "project submissions" in chunk["text"].lower() else 0.0
    return round((coverage * 0.55) + (direct_coverage * 0.35) + phrase_bonus, 4)


def run_reranking() -> tuple[list[dict], list[dict]]:
    candidates = retrieve(QUERY, k=CANDIDATE_K, chunk_size=12)
    reranked = [
        {**candidate, "rerank_score": rerank_score(QUERY, candidate)}
        for candidate in candidates
    ]
    reranked.sort(
        key=lambda item: (-item["rerank_score"], -item["score"], item["metadata"]["source"])
    )
    return candidates, reranked[:FINAL_K]


def _rows(label: str, chunks: list[dict], include_rerank: bool = False) -> list[str]:
    lines = [f"## {label}", "", "| Rank | Vector score | Re-rank score | Source | Metadata | Text |", "| ---: | ---: | ---: | --- | --- | --- |"]
    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        rerank = f"{chunk['rerank_score']:.4f}" if include_rerank else "n/a"
        text = chunk["text"].replace("|", "\\|")
        lines.append(
            f"| {rank} | {chunk['score']:.4f} | {rerank} | `{metadata['source']}` | "
            f"doc_type={metadata['doc_type']}, chunk_index={metadata['chunk_index']} | {text} |"
        )
    return lines


def write_report(candidates: list[dict], final_chunks: list[dict]) -> None:
    lines = [
        "# Chunk Re-Ranking Results",
        "",
        f"**Query:** {QUERY}  ",
        f"**Candidate set:** {len(candidates)} chunks retrieved (up to {CANDIDATE_K} requested)  ",
        f"**Final context:** top {len(final_chunks)} chunks after re-ranking",
        "",
        "The first stage uses the existing vector-style lexical score. The second "
        "stage combines query coverage, coverage of direct task terms, and a phrase "
        "match for `project submissions`.",
        "",
    ]
    lines.extend(_rows("Before Re-Ranking: Initial Candidate Order", candidates))
    lines.extend(["", *_rows("After Re-Ranking: Final Selected Context", final_chunks, True)])
    lines.extend(
        [
            "",
            "## Trade-Off",
            "",
            f"Requesting up to {CANDIDATE_K} candidates instead of {FINAL_K} gives the second "
            "stage more opportunities to recover a precise chunk, but it increases "
            "candidate transfer and scoring work. This offline scorer performs one "
            f"cheap pass over {len(candidates)} chunks; an LLM or cross-encoder would "
            "add model calls, latency, and per-candidate cost. The measured benefit "
            "here is precision-oriented ordering, while the candidate count keeps "
            "the extra work bounded.",
            "",
        ]
    )
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    initial, final = run_reranking()
    write_report(initial, final)
    print(f"Candidates: {len(initial)}; final chunks: {len(final)}")
    print(f"Report written to {OUTPUT_PATH}")