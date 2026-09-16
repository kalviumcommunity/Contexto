"""Generate and store embeddings for Contexto chunks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - used in offline demo environments
    OpenAI = Any  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def resolve_model() -> str:
    """Return the embedding model name from the environment or a default."""
    return os.getenv("EMBED_MODEL") or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"


def create_client() -> OpenAI:
    """Create the OpenAI-compatible client using environment configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in .env or your shell before generating embeddings."
        )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def build_records(chunks: list[dict], vectors: list[list[float]]) -> list[dict]:
    """Attach each vector to the chunk text and metadata that produced it."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match vector count ({len(vectors)})."
        )

    records: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        if "text" not in chunk:
            raise ValueError("Each chunk must include a 'text' field.")
        records.append(
            {
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "embedding": list(vector),
            }
        )
    return records


def embed_chunks(
    chunks: list[dict],
    *,
    model: str | None = None,
    client: OpenAI | None = None,
) -> list[dict]:
    """Send a batch of chunks to the embedding API and return vector records."""
    if not chunks:
        return []

    if client is None:
        client = create_client()
    model_name = model or resolve_model()

    response = client.embeddings.create(
        model=model_name,
        input=[chunk["text"] for chunk in chunks],
    )

    embeddings = [item.embedding for item in response.data]
    return build_records(chunks, embeddings)


def sample_chunks() -> list[dict]:
    """Prepare a small corpus matching the metadata model used elsewhere in the project."""
    return [
        {
            "text": "Password reset instructions for learner accounts.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 0,
                "section": "Overview",
                "page": 1,
            },
        },
        {
            "text": "Learners can recover access using their registered email.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 1,
                "section": "Overview",
                "page": 1,
            },
        },
        {
            "text": "Support agents can verify a learner identity before issuing a temporary password.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 2,
                "section": "Support",
                "page": 2,
            },
        },
    ]


def demo_vectors(chunks: list[dict], dim: int = 6) -> list[list[float]]:
    """Create deterministic fallback vectors for offline or demo runs."""
    vectors: list[list[float]] = []
    for index, chunk in enumerate(chunks):
        base = sum(ord(char) for char in chunk["text"]) % 97
        vector = []
        for offset in range(dim):
            value = ((base + index * 11 + offset * 7) % 100) / 100
            vector.append(round(value, 6))
        vectors.append(vector)
    return vectors


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors, where higher is more similar."""
    left = [float(value) for value in a]
    right = [float(value) for value in b]

    if len(left) != len(right):
        raise ValueError(f"Vector lengths differ ({len(left)} vs {len(right)}).")
    if not left or not right:
        return 0.0

    dot_product = sum(x * y for x, y in zip(left, right))
    left_norm = (sum(x * x for x in left)) ** 0.5
    right_norm = (sum(x * x for x in right)) ** 0.5

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def rank_chunks_for_query(query_vector: Sequence[float], records: list[dict]) -> list[dict]:
    """Score each chunk against a query vector and return them from most similar to least."""
    ranked = []
    for record in records:
        score = cosine_similarity(query_vector, record["embedding"])
        ranked.append({**record, "score": round(score, 6)})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def batches(items: Sequence[Any], size: int):
    """Yield consecutive batches of a fixed maximum size."""
    if size <= 0:
        raise ValueError("Batch size must be greater than zero.")
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def estimate_tokens(texts: Sequence[str]) -> int:
    """Estimate token counts using a simple word-based heuristic for run summaries."""
    total = 0
    for text in texts:
        words = (text or "").split()
        if not words:
            continue
        total += len(words)
    return total


def embed_with_retry(client: Any, model: str, texts: Sequence[str], *, max_attempts: int = 5, base_wait_seconds: float = 1.0):
    """Retry transient embedding failures with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return client.embeddings.create(model=model, input=list(texts))
        except Exception as error:  # pragma: no cover - exercised via integration path at runtime
            if attempt == max_attempts - 1:
                raise
            wait_seconds = base_wait_seconds * (2 ** attempt)
            print(f"retrying after error: {error} | wait={wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError("Embedding failed after retries.")


def run_batch_embedding(
    chunks: list[dict],
    *,
    client: Any | None = None,
    model: str | None = None,
    batch_size: int = 64,
    existing_ids: set[str] | None = None,
    price_per_1k_tokens: float = 0.00002,
    max_attempts: int = 5,
    base_wait_seconds: float = 1.0,
) -> tuple[list[dict], dict]:
    """Embed only pending chunks in batches and return records plus a cost/throughput summary."""
    if client is None:
        client = create_client()
    model_name = model or resolve_model()
    existing = set(existing_ids or [])
    pending = []
    for chunk in chunks:
        chunk_id = chunk.get("id") or chunk.get("metadata", {}).get("id") or chunk.get("metadata", {}).get("chunk_index")
        if chunk_id in existing:
            continue
        pending.append(chunk)

    summary = {
        "total_chunks": len(chunks),
        "skipped_existing": len(chunks) - len(pending),
        "embedded": 0,
        "failed": 0,
        "input_tokens": 0,
        "attempted_batches": 0,
        "estimated_cost_usd": 0.0,
    }

    records: list[dict] = []
    for batch in batches(pending, batch_size):
        summary["attempted_batches"] += 1
        texts = [chunk["text"] for chunk in batch]
        summary["input_tokens"] += estimate_tokens(texts)
        try:
            response = embed_with_retry(client, model_name, texts, max_attempts=max_attempts, base_wait_seconds=base_wait_seconds)
            batch_records = build_records(batch, [item.embedding for item in response.data])
            records.extend(batch_records)
            summary["embedded"] += len(batch_records)
        except Exception as error:
            summary["failed"] += len(batch)
            print(f"Batch failed after retries: {error}")

    summary["estimated_cost_usd"] = summary["input_tokens"] / 1000 * price_per_1k_tokens
    return records, summary


def render_batch_summary(summary: dict, *, model: str) -> str:
    """Render a markdown summary for a batch embedding run."""
    lines = [
        "# Batch Embedding Summary",
        "",
        f"- Model: {model}",
        f"- Total chunks: {summary['total_chunks']}",
        f"- Skipped existing: {summary['skipped_existing']}",
        f"- Embedded: {summary['embedded']}",
        f"- Failed: {summary['failed']}",
        f"- Input tokens: {summary['input_tokens']}",
        f"- Estimated cost (USD): ${summary['estimated_cost_usd']:.8f}",
        f"- Attempted batches: {summary['attempted_batches']}",
        "",
        "## Notes",
        "",
        "- Batching reduces per-request overhead.",
        "- Exponential backoff helps absorb rate-limit and transient errors.",
        "- Re-runs skip chunks already seen so the pipeline avoids duplicate costs.",
    ]
    return "\n".join(lines)


def render_output(records: list[dict], model: str) -> str:
    """Generate a markdown report showing stored text, metadata, and vector previews."""
    vector_length = len(records[0]["embedding"]) if records else 0
    sample_values = [round(value, 6) for value in records[0]["embedding"][:5]] if records else []

    lines = [
        "# Embedding Results",
        "",
        f"- Model: {model}",
        f"- Chunks embedded: {len(records)}",
        f"- Vector length: {vector_length}",
        f"- Sample values: {sample_values}",
        "",
    ]

    for index, record in enumerate(records):
        lines.extend(
            [
                f"## Chunk {index}",
                "",
                "```json",
                json.dumps(
                    {
                        "text": record["text"],
                        "metadata": record["metadata"],
                        "embedding_preview": [round(value, 6) for value in record["embedding"][:5]],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def render_similarity_report(query: str, query_vector: Sequence[float], ranked: list[dict]) -> str:
    """Generate a markdown report showing the strongest and weakest retrieval matches."""
    lines = [
        "# Similarity Ranking Results",
        "",
        f"- Query: {query}",
        f"- Query vector: {[round(value, 6) for value in query_vector]}",
        "",
        "## Ranked chunks",
        "",
    ]

    for index, item in enumerate(ranked, start=1):
        lines.extend(
            [
                f"### Rank {index}: score={item['score']:.6f}",
                "",
                "```json",
                json.dumps(
                    {
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "score": item["score"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Higher cosine similarity scores indicate a closer semantic match.",
            "- The top-ranked chunk is the best retrieval candidate for the query.",
            "- Lower scores still provide useful negatives for comparing relevance and retrieval quality.",
        ]
    )
    return "\n".join(lines)


def run_quality_checks(test_cases: list[dict], records: list[dict]) -> list[dict]:
    """Run known query-to-source checks and report whether the expected source ranks highest."""
    report = []
    for case in test_cases:
        query_vector = case.get("query_vector")
        if query_vector is None:
            raise ValueError(f"Each test case must include a 'query_vector': {case}")
        ranked = rank_chunks_for_query(query_vector, records)
        top = ranked[0]
        passed = top["metadata"].get("source") == case["expected_source"]
        report.append(
            {
                "query": case["query"],
                "expected_source": case["expected_source"],
                "top_source": top["metadata"].get("source"),
                "top_score": round(float(top["score"]), 6),
                "passed": passed,
                "note": "expected source ranked on top" if passed else "unexpected top source; investigate model or chunk mismatch",
            }
        )
    return report


def render_sanity_report(report: list[dict]) -> str:
    """Render a markdown sanity report summarising query quality checks."""
    passed = sum(1 for row in report if row["passed"])
    failed = len(report) - passed
    lines = [
        "# Sanity Report",
        "",
        f"- Tests: {len(report)}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        "",
        "## Results",
        "",
    ]

    for index, row in enumerate(report, start=1):
        lines.extend(
            [
                f"### Test {index}",
                "",
                "```json",
                json.dumps(
                    {
                        "query": row["query"],
                        "expected_source": row["expected_source"],
                        "top_source": row["top_source"],
                        "top_score": row["top_score"],
                        "passed": row["passed"],
                        "note": row["note"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Summary",
            "",
            "summary: related chunks should rank above unrelated ones for a trustworthy retrieval pipeline.",
            "summary: if a query fails, inspect model consistency, vector alignment, text cleaning, and metadata mapping.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings for Contexto chunks.")
    parser.add_argument("--demo", action="store_true", help="Use the offline demo corpus without requiring API credentials.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "embedding_results.md", help="Location for the saved sample output.")
    parser.add_argument("--model", default=None, help="Override the embedding model name.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the demo embedding pipeline.")
    args = parser.parse_args()

    model_name = args.model or resolve_model()
    chunks = sample_chunks()

    if args.demo or not os.getenv("OPENAI_API_KEY"):
        print("Using demo embedding vectors because OPENAI_API_KEY is not configured.")

        class DemoEmbeddingClient:
            @property
            def embeddings(self):
                return self

            def create(self, model, input):
                class FakeResponse:
                    def __init__(self, vectors):
                        self.data = [type("Item", (), {"embedding": vector})() for vector in vectors]
                return FakeResponse(demo_vectors([{"text": text} for text in input], dim=6))

        demo_client = DemoEmbeddingClient()
        records, summary = run_batch_embedding(
            [{**chunk, "id": f"{chunk['metadata']['source']}-{chunk['metadata'].get('chunk_index', idx)}"} for idx, chunk in enumerate(chunks)],
            client=demo_client,
            model=model_name,
            batch_size=args.batch_size,
            existing_ids={"account-guide.md-1"},
            price_per_1k_tokens=0.00002,
        )
    else:
        client = create_client()
        records, summary = run_batch_embedding(chunks, client=client, model=model_name, batch_size=args.batch_size, price_per_1k_tokens=0.00002)

    print(f"model: {model_name}")
    print(f"records: {len(records)}")
    print(f"vector length: {len(records[0]['embedding'])}")
    print(f"sample values: {records[0]['embedding'][:5]}")
    print("batch summary:", summary)

    query = "How can a learner reset their password?"
    query_vector = demo_vectors([{"text": query}], dim=len(records[0]["embedding"]))[0]
    ranked = rank_chunks_for_query(query_vector, records)
    print("Top match:", ranked[0]["score"], ranked[0]["text"])
    print("Bottom match:", ranked[-1]["score"], ranked[-1]["text"])

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_output(records, model_name), encoding="utf-8")
    print(f"Saved sample output: {output_path}")

    ranking_path = PROJECT_ROOT / "outputs" / "similarity_ranking_results.md"
    ranking_path.write_text(render_similarity_report(query, query_vector, ranked), encoding="utf-8")
    print(f"Saved ranking output: {ranking_path}")

    batch_summary_path = PROJECT_ROOT / "outputs" / "batch_embedding_results.md"
    batch_summary_path.write_text(render_batch_summary(summary, model=model_name), encoding="utf-8")
    print(f"Saved batch summary: {batch_summary_path}")

    sanity_cases = [
        {
            "query": "How can a learner reset their password?",
            "expected_source": "account-guide.md",
            "query_vector": demo_vectors([{"text": "How can a learner reset their password?"}], dim=len(records[0]["embedding"]))[0],
        },
        {
            "query": "When does the cafeteria menu change?",
            "expected_source": "campus-guide.md",
            "query_vector": demo_vectors([{"text": "When does the cafeteria menu change?"}], dim=len(records[0]["embedding"]))[0],
        },
    ]
    sanity_report = run_quality_checks(sanity_cases, records)
    sanity_path = PROJECT_ROOT / "outputs" / "sanity_report.md"
    sanity_path.write_text(render_sanity_report(sanity_report), encoding="utf-8")
    print(f"Saved sanity report: {sanity_path}")


if __name__ == "__main__":
    main()
