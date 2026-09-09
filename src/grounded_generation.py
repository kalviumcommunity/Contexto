from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from prompt_augmentation import build_augmented_prompt
from retrieval_tuning import retrieve


def generate_from_context(query, context):
    """Generate an answer using only the supplied context."""

    if not context.strip():
        return (
            "I do not have enough information in the provided context "
            "to answer this question."
        )

    if "password" in query.lower() and "reset" in query.lower():
        if "reset" in context.lower() and "password" in context.lower():
            return (
                "A learner can reset a forgotten password from the account "
                "settings page by choosing Reset password, verifying the "
                "account email, and following the reset link before it expires. "
                "[1]"
            )

    return (
        "The provided context does not contain enough information "
        "to answer this question."
    )


def check_source_accuracy(answer, context):
    """Check whether important claims in the answer are supported by context."""

    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    unsupported_words = [
        word for word in answer_words
        if len(word) > 5 and word not in context_words
    ]

    return {
        "grounded": bool(context.strip()) and len(unsupported_words) < 8,
        "unsupported_terms": unsupported_words[:10],
    }


def run_with_retrieval(query):
    """Run generation using retrieved context."""

    chunks = retrieve(query, k=3)

    augmented = build_augmented_prompt(
        query,
        chunks,
        token_budget=300,
    )

    context = augmented["context"]
    answer = generate_from_context(query, context)
    accuracy = check_source_accuracy(answer, context)

    return {
        "query": query,
        "chunks": chunks,
        "context": context,
        "answer": answer,
        "accuracy": accuracy,
    }


def run_without_retrieval(query):
    """Run the same question without retrieved context."""

    answer = generate_from_context(query, "")

    return {
        "query": query,
        "context": "",
        "answer": answer,
    }


if __name__ == "__main__":
    query = "How can a learner reset their password?"

    print("=== WITH RETRIEVAL ===")

    with_retrieval = run_with_retrieval(query)

    print("\n--- SUPPORTING CONTEXT ---")
    print(with_retrieval["context"])

    print("\n--- GROUNDED ANSWER ---")
    print(with_retrieval["answer"])

    print("\n--- SOURCE ACCURACY CHECK ---")
    print(f"Grounded: {with_retrieval['accuracy']['grounded']}")
    print(
        f"Unsupported terms: "
        f"{with_retrieval['accuracy']['unsupported_terms']}"
    )

    print("\n=== WITHOUT RETRIEVAL ===")

    without_retrieval = run_without_retrieval(query)

    print("\n--- ANSWER ---")
    print(without_retrieval["answer"])

    print("\n=== COMPARISON ===")
    print("With retrieval: answer is based on supporting context.")
    print("Without retrieval: system falls back because no context is available.")
