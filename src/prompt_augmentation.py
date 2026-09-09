from typing import List, Dict


DEFAULT_TOKEN_BUDGET = 300
RESERVED_TOKENS = 100


def estimate_tokens(text: str) -> int:
    """Estimate token usage using a simple word-based approximation."""
    if not text:
        return 0

    return max(1, len(text.split()))


def format_chunk(chunk: Dict, source_number: int) -> str:
    """Format one retrieved chunk with a source marker."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "unknown")
    text = chunk.get("text", "").strip()

    return f"[{source_number}] {source}\n{text}"


def assemble_augmented_context(
    chunks: List[Dict],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    reserved_tokens: int = RESERVED_TOKENS,
) -> Dict:
    """Inject retrieved chunks while respecting the available token budget."""

    available_tokens = max(0, token_budget - reserved_tokens)

    selected_chunks = []
    formatted_chunks = []
    tokens_used = 0

    for index, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(chunk, index)
        chunk_tokens = estimate_tokens(formatted)

        if tokens_used + chunk_tokens > available_tokens:
            break

        formatted_chunks.append(formatted)
        selected_chunks.append(chunk)
        tokens_used += chunk_tokens

    context = "\n\n".join(formatted_chunks)

    return {
        "context": context,
        "selected_chunks": selected_chunks,
        "tokens_used": tokens_used,
        "available_tokens": available_tokens,
        "token_budget": token_budget,
        "reserved_tokens": reserved_tokens,
    }


def build_augmented_prompt(
    query: str,
    chunks: List[Dict],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Dict:
    """Build a grounded prompt containing instructions, context and query."""

    result = assemble_augmented_context(
        chunks,
        token_budget=token_budget,
    )

    instructions = """You are a grounded question-answering assistant.

Answer the user's question ONLY using the information provided in the
context below.

Do not use outside knowledge or invent information.

If the context does not contain enough information to answer the question,
clearly say that the provided context is insufficient.

When possible, reference the supporting source using its source marker
such as [1] or [2].
"""

    prompt = (
        f"{instructions}\n"
        f"CONTEXT:\n"
        f"{result['context']}\n\n"
        f"USER QUESTION:\n"
        f"{query}\n"
    )

    result["prompt"] = prompt

    return result


if __name__ == "__main__":
    sample_chunks = [
        {
            "text": (
                "Learners can reset a forgotten password from the account "
                "settings page. Choose Reset password, verify the account "
                "email, and follow the link before it expires."
            ),
            "metadata": {"source": "account-guide.md"},
            "score": 0.571,
        },
        {
            "text": (
                "Learners should keep their account email accessible when "
                "performing account recovery."
            ),
            "metadata": {"source": "news-brief.txt"},
            "score": 0.250,
        },
        {
            "text": (
                "Submissions should follow the instructions provided in "
                "the relevant learner documentation."
            ),
            "metadata": {"source": "submission-rubric.md"},
            "score": 0.143,
        },
    ]

    query = "How can a learner reset their password?"

    result = build_augmented_prompt(
        query,
        sample_chunks,
        token_budget=300,
    )

    print("=== AUGMENTED PROMPT ===")
    print(result["prompt"])

    print("=== TOKEN BUDGET ===")
    print(f"Total budget: {result['token_budget']}")
    print(f"Reserved tokens: {result['reserved_tokens']}")
    print(f"Context budget: {result['available_tokens']}")
    print(f"Context tokens used: {result['tokens_used']}")
    print(f"Chunks injected: {len(result['selected_chunks'])}")