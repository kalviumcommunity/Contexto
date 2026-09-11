import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import Settings, create_app, load_settings


def test_load_settings_uses_environment_and_defaults():
    settings = load_settings({
        "OPENAI_API_KEY": "test-key",
        "VECTOR_DB_URL": "http://vector-db",
    })

    assert settings.openai_api_key == "test-key"
    assert settings.vector_db_url == "http://vector-db"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.collection_name == "rag_chunks"


def test_load_settings_rejects_missing_required_values():
    try:
        load_settings({"OPENAI_API_KEY": "test-key"})
    except ValueError as error:
        assert "VECTOR_DB_URL" in str(error)
    else:
        raise AssertionError("missing VECTOR_DB_URL should fail")


def test_query_returns_structured_grounded_answer():
    def pipeline(question):
        assert question == "What evidence is required?"
        return {
            "answer": "A PR link is required.",
            "sources": [{"source": "submission-rubric.md", "chunk_id": "chunk-2", "score": 0.84}],
            "status": "answered",
        }

    client = TestClient(create_app(
        rag_pipeline=pipeline,
        settings=Settings("key", "embed-model", "http://db", "chunks"),
    ))
    response = client.post("/query", json={"question": "What evidence is required?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "A PR link is required.",
        "sources": [{"source": "submission-rubric.md", "chunk_id": "chunk-2", "score": 0.84}],
        "status": "answered",
    }


def test_query_validates_question_length():
    client = TestClient(create_app(settings=Settings("key", "embed", "http://db", "chunks")))

    response = client.post("/query", json={"question": "?"})

    assert response.status_code == 422


def test_query_returns_500_without_leaking_pipeline_error():
    def pipeline(_question):
        raise RuntimeError("secret backend detail")

    client = TestClient(create_app(
        rag_pipeline=pipeline,
        settings=Settings("key", "embed", "http://db", "chunks"),
    ))
    response = client.post("/query", json={"question": "Explain the evidence"})

    assert response.status_code == 500
    assert response.json() == {"detail": "RAG service failed"}