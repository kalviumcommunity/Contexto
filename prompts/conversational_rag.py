"""Conversation-aware retrieval and grounded answer orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from prompts.guardrails import (
    MIN_SUPPORTING_CHUNKS,
    MIN_TOP_SCORE,
    REFUSAL_MESSAGE,
    retrieval_is_strong,
)


HistoryMessage = dict[str, str]
Chunk = dict[str, Any]


def rewrite_followup(
    history: Sequence[HistoryMessage],
    question: str,
    call_llm: Callable[[str], str],
) -> str:
    """Rewrite a follow-up as a standalone retrieval query."""
    history_text = "\n".join(
        f"{message['role']}: {message['content']}" for message in history
    )
    prompt = (
        "Rewrite the user's latest question as a standalone search query. "
        "Use the conversation history only to resolve references. "
        "Do not answer the question.\n\n"
        f"History:\n{history_text}\n\n"
        f"Latest question:\n{question}"
    )
    rewritten = call_llm(prompt).strip()
    if not rewritten:
        raise ValueError("the rewritten query cannot be empty")
    return rewritten


def conversational_answer(
    history: list[HistoryMessage],
    user_question: str,
    rewrite_fn: Callable[[Sequence[HistoryMessage], str], str],
    retrieve_fn: Callable[[str], Sequence[Chunk]],
    generate_fn: Callable[[str, Sequence[Chunk]], dict[str, Any]],
    *,
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> dict[str, Any]:
    """Answer a question using rewritten retrieval and fresh evidence."""
    standalone_query = rewrite_fn(history, user_question)
    chunks = list(retrieve_fn(standalone_query))

    if retrieval_is_strong(
        chunks,
        min_top_score=min_top_score,
        min_supporting_chunks=min_supporting_chunks,
    ):
        answer_result = generate_fn(user_question, chunks)
        answer = answer_result["answer"]
        status = "answered"
    else:
        answer_result = {"answer": REFUSAL_MESSAGE, "sources": []}
        answer = REFUSAL_MESSAGE
        status = "refused_weak_context"

    history.extend(
        [
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": answer},
        ]
    )

    return {
        **answer_result,
        "rewritten_query": standalone_query,
        "retrieved_chunks": chunks,
        "sources": [chunk.get("metadata", {}) for chunk in chunks]
        if status == "answered"
        else [],
        "status": status,
    }