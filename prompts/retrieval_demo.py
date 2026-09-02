"""Demonstration of similarity search and top-k retrieval for the Contexto RAG system.

This module shows how to:
1. Embed a user query using the same model as the indexed documents
2. Search the vector database for top-k similar chunks
3. Inspect retrieved results with similarity scores and metadata
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from vector_indexing import retrieve, retrieve_with_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a list of text strings."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def demonstrate_retrieval() -> None:
    """Show how to embed queries and retrieve top-k similar chunks."""
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb is required for retrieval demo") from exc

    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY was not found in .env")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Set up the vector database collection
    chroma_client = chromadb.PersistentClient(path="./.chroma")
    collection = chroma_client.get_or_create_collection("contexto_chunks")

    # Sample indexed chunks (these would normally come from your document ingestion)
    sample_chunks = [
        {
            "id": "article-1:chunk-0",
            "embedding": embed_texts(client, [
                "Journalists must verify all facts against original source documentation."
            ])[0],
            "text": "Journalists must verify all facts against original source documentation.",
            "metadata": {
                "source": "journalism_guide.txt",
                "chunk_index": 0,
                "section": "Verification",
            },
        },
        {
            "id": "article-1:chunk-1",
            "embedding": embed_texts(client, [
                "A credible citation requires the source name, publication date, and exact quote."
            ])[0],
            "text": "A credible citation requires the source name, publication date, and exact quote.",
            "metadata": {
                "source": "journalism_guide.txt",
                "chunk_index": 1,
                "section": "Citations",
            },
        },
        {
            "id": "article-1:chunk-2",
            "embedding": embed_texts(client, [
                "Historical context helps readers understand the significance of current events."
            ])[0],
            "text": "Historical context helps readers understand the significance of current events.",
            "metadata": {
                "source": "journalism_guide.txt",
                "chunk_index": 2,
                "section": "Context",
            },
        },
    ]

    print("\n=== Indexing sample chunks ===")
    for chunk in sample_chunks:
        collection.upsert(
            ids=[chunk["id"]],
            embeddings=[chunk["embedding"]],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
    print(f"Indexed {len(sample_chunks)} chunks")

    print("\n=== Demonstrating Top-K Retrieval ===")

    query = "How can a journalist verify their sources?"
    print(f"\nQuery: {query!r}\n")

    # Embed the query using the same model as the documents
    query_embedding = embed_texts(client, [query])[0]

    # Demonstrate different k values
    for k in [1, 3]:
        print(f"\n--- Results for k={k} ---")
        results = retrieve(collection, query_embedding, k=k)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank {rank}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Source: {result['metadata']['source']}")
            print(f"  Chunk Index: {result['metadata']['chunk_index']}")
            print(f"  Section: {result['metadata'].get('section', 'N/A')}")
            print(f"  Text: {result['text']}")

    print("\n=== Top-K Trade-Offs ===")
    print("Smaller k (e.g., 1-3):")
    print("  + Faster retrieval")
    print("  + Less context window consumed")
    print("  - May miss relevant information")
    print("\nLarger k (e.g., 5-10):")
    print("  + Better recall (more potentially relevant chunks)")
    print("  - Higher latency and cost")
    print("  - Risk of including irrelevant noise in the prompt")
    print("\nTune k based on:")
    print("  - Chunk size and quality")
    print("  - Question complexity")
    print("  - Available context window in your LLM")


if __name__ == "__main__":
    demonstrate_retrieval()
