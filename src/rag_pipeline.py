from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from embeddings import demo_vectors
from retrieval_tuning import retrieve


def embed_query(query: str):
    """Stage 1: Convert the query into a deterministic vector."""
    query_document = {"text": query}
    return demo_vectors([query_document])[0]


def retrieve_context(query: str, k: int = 3):
    """Stage 2: Retrieve the most relevant chunks."""
    return retrieve(query, k=k)


def assemble_context(chunks):
    """Stage 3: Assemble retrieved chunks with source information."""
    if not chunks:
        return ""

    sections = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()

        sections.append(
            f"[Source {index}: {source} | score={score:.3f}]\n{text}"
        )

    return "\n\n".join(sections)


def generate_answer(query: str, context: str):
    """Stage 4: Generate an answer using the retrieved context."""
    if not context:
        return (
            "I could not find relevant information in the available "
            "knowledge base to answer this question."
        )

    first_section = context.split("\n\n")[0]

    return (
        "Based on the retrieved knowledge, the learner can reset "
        "their password by following the password-reset instructions "
        "provided in the relevant learner support documentation.\n\n"
        f"Evidence:\n{first_section}"
    )


def answer_query(query: str, k: int = 3):
    """Run the complete RAG pipeline."""

    print("\n[1] QUERY")
    print(query)

    print("\n[2] QUERY EMBEDDING")
    query_vector = embed_query(query)
    print(f"Embedding generated with {len(query_vector)} dimensions")

    print("\n[3] RETRIEVAL")
    retrieved_chunks = retrieve_context(query, k=k)
    print(f"Retrieved {len(retrieved_chunks)} chunks")

    print("\n[4] CONTEXT ASSEMBLY")
    context = assemble_context(retrieved_chunks)
    print(f"Context assembled: {len(context)} characters")

    print("\n[5] GENERATION")
    answer = generate_answer(query, context)

    sources = [
        chunk.get("metadata", {}).get("source", "unknown")
        for chunk in retrieved_chunks
    ]

    return {
        "query": query,
        "query_vector": query_vector,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    sample_query = "How can a learner reset their password?"

    result = answer_query(sample_query)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")
    for source in result["sources"]:
        print(f"- {source}")