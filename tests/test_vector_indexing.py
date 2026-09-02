import pytest

from prompts.vector_indexing import (
    batches,
    _normalize_search_result,
    retrieve,
    to_vector_record,
)


def test_to_vector_record_keeps_text_and_metadata():
    chunk = {
        "id": "doc-1:0",
        "embedding": [0.1, 0.2, 0.3],
        "text": "This is a sample chunk.",
        "metadata": {
            "source": "sample.txt",
            "chunk_index": 1,
            "section": "Intro",
        },
    }

    record = to_vector_record(chunk)

    assert record["id"] == "doc-1:0"
    assert record["vector"] == [0.1, 0.2, 0.3]
    assert record["text"] == "This is a sample chunk."
    assert record["metadata"] == {
        "source": "sample.txt",
        "chunk_index": 1,
        "section": "Intro",
    }


def test_batches_splits_items_into_fixed_sizes():
    items = list(range(5))

    assert list(batches(items, 2)) == [[0, 1], [2, 3], [4]]


def test_normalize_search_result_handles_dict_format():
    result = {
        "id": "doc-1:0",
        "score": 0.95,
        "text": "Sample text",
        "metadata": {"source": "article.txt", "chunk_index": 0},
    }

    normalized = _normalize_search_result(result)

    assert normalized == result


def test_normalize_search_result_handles_tuple_format():
    result = ("doc-1:0", 0.95, {"source": "article.txt"}, "Sample text")

    normalized = _normalize_search_result(result)

    assert normalized["id"] == "doc-1:0"
    assert normalized["score"] == 0.95
    assert normalized["text"] == "Sample text"
    assert normalized["metadata"] == {"source": "article.txt"}


def test_retrieve_raises_error_for_invalid_k():
    mock_collection = {}

    with pytest.raises(ValueError, match="k must be greater than 0"):
        retrieve(mock_collection, [0.1, 0.2, 0.3], k=0)

    with pytest.raises(ValueError, match="k must be greater than 0"):
        retrieve(mock_collection, [0.1, 0.2, 0.3], k=-1)


def test_retrieve_normalizes_chroma_response():
    """Test that retrieve can handle Chroma's dict response format."""
    mock_collection = type("Collection", (), {
        "query": lambda self, **kwargs: {
            "ids": [["doc-1:0", "doc-1:1"]],
            "distances": [[0.05, 0.15]],
            "documents": [["First chunk", "Second chunk"]],
            "metadatas": [[
                {"source": "doc1.txt", "chunk_index": 0},
                {"source": "doc1.txt", "chunk_index": 1},
            ]],
        }
    })()

    results = retrieve(mock_collection, [0.1, 0.2, 0.3], k=2)

    assert len(results) == 2
    assert results[0]["score"] == 0.05
    assert results[0]["text"] == "First chunk"
    assert results[0]["metadata"]["source"] == "doc1.txt"
    assert results[1]["score"] == 0.15
    assert results[1]["text"] == "Second chunk"


def test_retrieve_respects_k_parameter():
    """Test that retrieve passes the k parameter to the collection."""
    call_log = []
    
    def mock_query(self, **kwargs):
        call_log.append(kwargs)
        return {
            "ids": [["doc-1:0", "doc-1:1"]],
            "distances": [[0.05, 0.15]],
            "documents": [["Chunk 1", "Chunk 2"]],
            "metadatas": [[
                {"source": "doc1.txt", "chunk_index": i}
                for i in range(2)
            ]],
        }
    
    mock_collection = type("Collection", (), {"query": mock_query})()
    
    retrieve(mock_collection, [0.1, 0.2, 0.3], k=5)
    
    # Verify that query was called with n_results parameter
    assert len(call_log) == 1
    assert call_log[0].get("n_results") == 5
