from prompts.guardrails import guarded_answer, retrieval_is_strong


def test_retrieval_is_strong_requires_a_high_scoring_chunk():
    assert retrieval_is_strong([{"score": 0.72}]) is True
    assert retrieval_is_strong([{"score": 0.71}]) is False
    assert retrieval_is_strong([]) is False


def test_guarded_answer_refuses_without_calling_generator():
    generated = False

    def retrieve(_question):
        return [{"score": 0.2, "text": "Unrelated context"}]

    def generate(_question, _chunks):
        nonlocal generated
        generated = True
        return {"answer": "unsupported", "sources": ["source.txt"]}

    result = guarded_answer("What is the refund policy?", retrieve, generate)

    assert result == {
        "answer": "I don't have enough reliable context to answer that.",
        "sources": [],
        "status": "refused_weak_context",
    }
    assert generated is False


def test_guarded_answer_generates_with_strong_context():
    chunks = [{"score": 0.91, "text": "Submission requires source citations."}]
    calls = []

    def retrieve(question):
        calls.append(("retrieve", question))
        return chunks

    def generate(question, retrieved_chunks):
        calls.append(("generate", question, retrieved_chunks))
        return {
            "answer": "Include source citations.",
            "sources": ["guide.txt"],
        }

    result = guarded_answer("What is required?", retrieve, generate)

    assert result == {
        "answer": "Include source citations.",
        "sources": ["guide.txt"],
        "status": "answered",
    }
    assert calls == [
        ("retrieve", "What is required?"),
        ("generate", "What is required?", chunks),
    ]