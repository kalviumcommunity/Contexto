"""Demonstration of metadata filtering and hybrid search for the Contexto RAG system.

This module shows how to:
1. Compare filtered and unfiltered retrieval results
2. Understand vector search vs. keyword search trade-offs
3. Use hybrid ranking to combine semantic and lexical matching
4. Tune weights for different use cases
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from vector_indexing import retrieve, retrieve_with_embedding, keyword_score, hybrid_rank

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


def show_results(label: str, results: list[dict]) -> None:
    """Pretty-print retrieval results with scores, sources, and text preview."""
    print(f"\n{label}")
    print("=" * 80)
    
    if not results:
        print("  (no results)")
        return
    
    for rank, item in enumerate(results, start=1):
        print(f"\nRank {rank}:")
        if "hybrid_score" in item:
            print(f"  Hybrid Score: {item['hybrid_score']:.4f}")
            print(f"  Vector Score: {item['score']:.4f}")
            print(f"  Keyword Score: {item['keyword_score']:.1f}")
        else:
            print(f"  Score: {item['score']:.4f}")
        
        metadata = item.get("metadata", {})
        print(f"  Source: {metadata.get('source', 'N/A')}")
        print(f"  Section: {metadata.get('section', 'N/A')}")
        print(f"  Chunk Index: {metadata.get('chunk_index', 'N/A')}")
        
        text = item.get("text", "")
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  Text: {preview}")


def demonstrate_metadata_filtering() -> None:
    """Show how metadata filters scope retrieval to relevant subsets."""
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb is required for filtering demo") from exc

    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY was not found in .env")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Set up the vector database collection
    chroma_client = chromadb.PersistentClient(path="./.chroma")
    collection = chroma_client.get_or_create_collection("contexto_filtering_demo")
    collection.delete_all()

    # Sample indexed chunks with diverse metadata
    sample_chunks = [
        {
            "id": "account-1:0",
            "embedding": embed_texts(client, [
                "To reset your password, click the forgot password link on the login page."
            ])[0],
            "text": "To reset your password, click the forgot password link on the login page.",
            "metadata": {
                "source": "faq.txt",
                "chunk_index": 0,
                "section": "Account access",
            },
        },
        {
            "id": "account-1:1",
            "embedding": embed_texts(client, [
                "Password requirements include at least 8 characters with uppercase and numbers."
            ])[0],
            "text": "Password requirements include at least 8 characters with uppercase and numbers.",
            "metadata": {
                "source": "faq.txt",
                "chunk_index": 1,
                "section": "Security policy",
            },
        },
        {
            "id": "account-1:2",
            "embedding": embed_texts(client, [
                "Two-factor authentication adds extra security to your account login process."
            ])[0],
            "text": "Two-factor authentication adds extra security to your account login process.",
            "metadata": {
                "source": "security_guide.txt",
                "chunk_index": 0,
                "section": "Security policy",
            },
        },
        {
            "id": "account-1:3",
            "embedding": embed_texts(client, [
                "Your password must be changed every 90 days for compliance."
            ])[0],
            "text": "Your password must be changed every 90 days for compliance.",
            "metadata": {
                "source": "compliance.txt",
                "chunk_index": 0,
                "section": "Password policy",
            },
        },
    ]

    print("\n" + "=" * 80)
    print("METADATA FILTERING & HYBRID SEARCH DEMONSTRATION")
    print("=" * 80)

    # Index all chunks
    print("\n[Indexing sample chunks with diverse metadata]")
    for chunk in sample_chunks:
        collection.upsert(
            ids=[chunk["id"]],
            embeddings=[chunk["embedding"]],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
    print(f"✓ Indexed {len(sample_chunks)} chunks")

    # Part 1: Unfiltered vs. Filtered Retrieval
    print("\n" + "=" * 80)
    print("PART 1: METADATA FILTERING")
    print("=" * 80)

    query = "How can a learner reset their password?"
    print(f"\nUser Query: {query!r}")

    query_embedding = embed_texts(client, [query])[0]

    # Unfiltered search
    unfiltered = retrieve(collection, query_embedding, k=3)
    show_results("Unfiltered Results (all sections)", unfiltered)

    # Filtered search - only "Account access" section
    filtered = retrieve(
        collection,
        query_embedding,
        k=3,
        metadata_filter={"section": "Account access"}
    )
    show_results("Filtered Results (section='Account access' only)", filtered)

    print("\nFiltering Impact:")
    print("  ✓ Filtered results focused on account access procedures")
    print("  ✓ Unfiltered included security policy which is less relevant")
    print("  ✓ Filtering improves precision for specific use cases")

    # Part 2: Vector vs. Keyword Search
    print("\n" + "=" * 80)
    print("PART 2: VECTOR vs. KEYWORD SEARCH")
    print("=" * 80)

    query2 = "password reset"
    print(f"\nUser Query: {query2!r}")

    query_embedding2 = embed_texts(client, [query2])[0]
    vector_results = retrieve(collection, query_embedding2, k=4)

    print("\nVector Search (Semantic Similarity):")
    print("  Finds chunks with related meaning, even if wording differs")
    show_results("Vector Results", vector_results)

    # Calculate keyword scores
    print("\nKeyword Search Analysis (Lexical Matching):")
    keywords = ["password", "reset"]
    for result in vector_results:
        lex_score = keyword_score(result["text"], keywords)
        print(f"  '{result['text'][:60]}...' → keyword_score={lex_score}")

    # Part 3: Hybrid Ranking
    print("\n" + "=" * 80)
    print("PART 3: HYBRID SEARCH (COMBINING VECTOR + KEYWORD)")
    print("=" * 80)

    print(f"\nQuery: {query2!r}")
    print("Keywords to boost: ['password', 'reset']")
    print("Weights: Vector=0.8, Keyword=0.2")

    hybrid_results = hybrid_rank(
        vector_results,
        keywords,
        vector_weight=0.8,
        keyword_weight=0.2
    )
    show_results("Hybrid Ranked Results", hybrid_results)

    # Show weight comparison
    print("\nHybrid Ranking Comparison:")
    print("  With vector_weight=0.8, keyword_weight=0.2:")
    print("    - Favors semantic similarity")
    print("    - Slight boost for exact keyword matches")
    print("\n  With vector_weight=0.5, keyword_weight=0.5:")
    hybrid_balanced = hybrid_rank(
        vector_results,
        keywords,
        vector_weight=0.5,
        keyword_weight=0.5
    )
    print("    - Balanced between semantics and exact matches")
    print(f"    - Top result: {hybrid_balanced[0]['text'][:60]}...")

    # Part 4: Combined Filtering + Hybrid
    print("\n" + "=" * 80)
    print("PART 4: COMBINED FILTERING + HYBRID SEARCH")
    print("=" * 80)

    print(f"\nQuery: {query2!r}")
    print("Filter: section='Password policy' or section='Account access'")
    print("Keywords: ['password', 'reset']")

    # Filtered retrieval
    filtered_results = retrieve(
        collection,
        query_embedding2,
        k=4,
        metadata_filter={"section": "Account access"}
    )

    # Apply hybrid ranking to filtered results
    combined = hybrid_rank(
        filtered_results,
        keywords,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    show_results("Filtered + Hybrid Results", combined)

    print("\nCombined Strategy Benefits:")
    print("  ✓ Filtering scopes to relevant sections first")
    print("  ✓ Hybrid ranking within scope maximizes precision")
    print("  ✓ More efficient and focused than unfiltered search")

    # Part 5: Use Case Recommendations
    print("\n" + "=" * 80)
    print("USE CASE RECOMMENDATIONS")
    print("=" * 80)

    print("""
✓ USE METADATA FILTERING WHEN:
  - Users have clear intent (e.g., "show only account-related docs")
  - Metadata reflects product structure (section, source, category)
  - Over-inclusive results waste context window
  - You want to ensure compliance (e.g., only public docs)

✓ USE KEYWORD/HYBRID MATCHING WHEN:
  - Users search for exact product names or IDs
  - Acronyms or codes are important (e.g., "SKU-12345")
  - Semantic similarity alone misses exact-match needs
  - You want to boost results containing specific terms

✓ USE COMBINED STRATEGY WHEN:
  - Large corpus with diverse metadata
  - Need both precision (filters) and recall (hybrid scoring)
  - Users expect relevant results from specific sections
  - Example: "Show password reset steps from account access section"
    """)

    collection.delete_all()


if __name__ == "__main__":
    demonstrate_metadata_filtering()
