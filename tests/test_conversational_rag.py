import pytest

from prompts.conversational_rag import conversational_answer, rewrite_followup


def test_rewrite_followup_uses_history_and_returns_standalone_query():
    prompts = []

    def call_llm(prompt):
        prompts.append(prompt)
        return "What video explanation is required for project submission?\n"

    history = [
        {"role": "user", "content": "What evidence is required for submission?"},
        {"role": "assistant", "content": "A PR link and video explanation."},
    ]

    result = rewrite_followup(history, "What about the video?", call_llm)

    assert result == "What video explanation is required for project submission?"
    assert "What about the video?" in prompts[0]
    assert "A PR link and video explanation." in prompts[0]


def test_rewrite_followup_rejects_empty_model_output():
    with pytest.raises(ValueError, match="rewritten query cannot be empty"):
        rewrite_followup([], "What about it?", lambda _prompt: "  ")


def test_conversational_answer_retrieves_with_rewrite_and_answers_original_question():
    history = [
        {"role": "user", "content": "What evidence is required?"},
        {"role": "assistant", "content": "A PR link and video explanation."},
    ]
    calls = []
    chunks = [
        {
            "score": 0.9,
            "text": "The submission requires a video explanation.",
            "metadata": {"source": "guide.txt", "chunk_index": 2},
        }
    ]

    def rewrite(history_arg, question):
        calls.append(("rewrite", list(history_arg), question))
        return "What video explanation is required for project submission?"

    def retrieve(query):
        calls.append(("retrieve", query))
        return chunks

    def generate(question, retrieved_chunks):
        calls.append(("generate", question, retrieved_chunks))
        return {"answer": "A video explanation is required.", "sources": []}

    result = conversational_answer(history, "What about the video?", rewrite, retrieve, generate)

    assert result == {
        "answer": "A video explanation is required.",
        "sources": [{"source": "guide.txt", "chunk_index": 2}],
        "rewritten_query": "What video explanation is required for project submission?",
        "retrieved_chunks": chunks,
        "status": "answered",
    }
    assert calls[1] == (
        "retrieve",
        "What video explanation is required for project submission?",
    )
    assert calls[2][0:2] == ("generate", "What about the video?")
    assert history[-2:] == [
        {"role": "user", "content": "What about the video?"},
        {"role": "assistant", "content": "A video explanation is required."},
    ]


def test_conversational_answer_refuses_weak_context_and_records_refusal():
    history = []
    generated = False

    def generate(_question, _chunks):
        nonlocal generated
        generated = True
        return {"answer": "unsupported", "sources": []}

    result = conversational_answer(
        history,
        "Does it apply to Sprint 2?",
        lambda _history, _question: "Sprint 2 submission requirements",
        lambda _query: [{"score": 0.2, "metadata": {"source": "unrelated.txt"}}],
        generate,
    )

    assert result["status"] == "refused_weak_context"
    assert result["answer"] == "I don't have enough reliable context to answer that."
    assert result["sources"] == []
    assert generated is False
    assert history[-1]["content"] == result["answer"]