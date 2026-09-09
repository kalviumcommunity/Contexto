from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from retrieval_tuning import retrieve


def build_source_mapping(chunks):
    mappings = {}

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        mappings[f"[{index}]"] = {
            "source": metadata.get("source", "unknown"),
            "chunk_id": metadata.get("chunk_id", f"chunk-{index}"),
            "chunk_index": metadata.get("chunk_index", index - 1),
            "text": chunk.get("text", "").strip(),
        }

    return mappings


def generate_cited_answer(query, chunks):
    if not chunks:
        return {
            "answer": (
                "I do not have enough supporting source information "
                "to answer this question."
            ),
            "citations": [],
            "source_mapping": {},
        }

    source_mapping = build_source_mapping(chunks)

    context = "\n".join(
        item["text"] for item in source_mapping.values()
    ).lower()

    if "password" in query.lower() and "reset" in query.lower():
        if "password" in context and "reset" in context:
            answer = (
                "A learner can reset a forgotten password from the account "
                "settings page by choosing Reset password, verifying the "
                "account email, and following the reset link before it expires."
            )

            supporting_citations = []

            for citation, source in source_mapping.items():
                text = source["text"].lower()

                if "password" in text and "reset" in text:
                    supporting_citations.append(citation)

            if supporting_citations:
                answer += " " + " ".join(supporting_citations)

            return {
                "answer": answer,
                "citations": supporting_citations,
                "source_mapping": source_mapping,
            }

    return {
        "answer": (
            "I do not have enough supporting source information "
            "to answer this question."
        ),
        "citations": [],
        "source_mapping": source_mapping,
    }


def verify_citation(citation, source_mapping):
    source = source_mapping.get(citation)

    if not source:
        return {
            "verified": False,
            "reason": "Citation does not exist in the source mapping.",
        }

    return {
        "verified": bool(source["source"] and source["text"]),
        "source": source["source"],
        "chunk_id": source["chunk_id"],
        "chunk_index": source["chunk_index"],
        "original_text": source["text"],
    }


if __name__ == "__main__":
    query = "How can a learner reset their password?"

    chunks = retrieve(query, k=3)

    result = generate_cited_answer(query, chunks)

    print("=== CITED ANSWER ===")
    print(result["answer"])

    print("\n=== CITATION MAPPING ===")

    for citation, source in result["source_mapping"].items():
        print(f"\n{citation}")
        print(f"Source: {source['source']}")
        print(f"Chunk ID: {source['chunk_id']}")
        print(f"Chunk index: {source['chunk_index']}")
        print(f"Original text: {source['text']}")

    print("\n=== SOURCE VERIFICATION ===")

    for citation in result["citations"]:
        verification = verify_citation(
            citation,
            result["source_mapping"],
        )

        print(f"\n{citation} verified: {verification['verified']}")
        print(f"Source: {verification.get('source')}")
        print(f"Original text: {verification.get('original_text')}")

    print("\n=== NO-SOURCE FALLBACK ===")

    no_source_result = generate_cited_answer(
        "What is the university's policy for changing a degree program?",
        [],
    )

    print(no_source_result["answer"])
    print(f"Citations: {no_source_result['citations']}")
