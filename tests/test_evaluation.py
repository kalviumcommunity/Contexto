from prompts.evaluation import evaluate_test_set, score_answer


TEST_SET = [
    {
        "question": "What evidence is required for project submission?",
        "expected_points": ["PR link", "sample output", "video explanation"],
        "expected_sources": {"submission-rubric.md"},
    },
    {
        "question": "What should the system do when context is missing?",
        "expected_points": ["refuse", "not enough information"],
        "expected_sources": {"guardrails.md"},
    },
]


def answer_fn(question):
    if question.startswith("What evidence"):
        return {
            "answer": "Provide a PR link, sample output, and video explanation.",
            "sources": [{"source": "submission-rubric.md"}],
            "retrieved_chunks": [{
                "text": "Submission requires a PR link, sample output, and video explanation.",
            }],
        }
    return {
        "answer": "Refuse and say there is not enough information.",
        "sources": [{"source": "guardrails.md"}],
        "retrieved_chunks": [{
            "text": "When context is missing, refuse and say there is not enough information.",
        }],
    }


def test_score_answer_marks_correct_grounded_and_cited_result():
    row = score_answer(TEST_SET[0], answer_fn)

    assert row["correctness"] == 1
    assert row["grounding"] == 1
    assert row["citation_accuracy"] == 1


def test_evaluate_test_set_summarizes_failures():
    def answer_with_bad_citation(question):
        result = dict(answer_fn(question))
        result["sources"] = [{"source": "wrong-source.md"}]
        return result

    summary = evaluate_test_set(TEST_SET, answer_with_bad_citation)

    assert summary["questions"] == 2
    assert summary["avg_correctness"] == 1.0
    assert summary["avg_grounding"] == 1.0
    assert summary["avg_citation_accuracy"] == 0.0
    assert len(summary["failures"]) == 2