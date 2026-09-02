import pytest

from prompts.vector_indexing import (
    batches,
    _normalize_search_result,
    retrieve,
    retrieve_with_embedding,
    keyword_score,
    hybrid_rank,
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


def test_retrieve_with_metadata_filter():
    """Test that retrieve passes metadata filters to the collection."""
    call_log = []
    
    def mock_query(self, **kwargs):
        call_log.append(kwargs)
        return {
            "ids": [["doc-1:0"]],
            "distances": [[0.05]],
            "documents": [["Filtered Chunk"]],
            "metadatas": [[
                {"source": "doc1.txt", "chunk_index": 0, "section": "Account access"}
            ]],
        }
    
    mock_collection = type("Collection", (), {"query": mock_query})()
    
    metadata_filter = {"section": "Account access"}
    retrieve(mock_collection, [0.1, 0.2, 0.3], k=3, metadata_filter=metadata_filter)
    
    # Verify that query was called with where parameter
    assert len(call_log) == 1
    assert call_log[0].get("where") == metadata_filter


def test_keyword_score_counts_occurrences():
    """Test that keyword_score counts keyword occurrences."""
    text = "Password reset requires password verification and password confirmation"
    keywords = ["password", "reset"]
    
    score = keyword_score(text, keywords)
    
    # "password" appears 3 times, "reset" appears 1 time = 4 total
    assert score == 4.0


def test_keyword_score_case_insensitive():
    """Test that keyword_score is case-insensitive."""
    text = "Password Reset is Required"
    keywords = ["password", "RESET"]
    
    score = keyword_score(text, keywords)
    
    assert score == 2.0


def test_keyword_score_empty_keywords():
    """Test that keyword_score returns 0 for empty keywords."""
    text = "Some text here"
    
    assert keyword_score(text, []) == 0.0


def test_keyword_score_no_matches():
    """Test that keyword_score returns 0 when keywords not found."""
    text = "The quick brown fox"
    keywords = ["password", "reset"]
    
    score = keyword_score(text, keywords)
    
    assert score == 0.0


def test_hybrid_rank_combines_scores():
    """Test that hybrid_rank combines vector and keyword scores."""
    vector_results = [
        {
            "score": 0.9,
            "text": "Password reset steps for account access",
            "metadata": {"source": "guide.txt"},
        },
        {
            "score": 0.7,
            "text": "How to verify your identity",
            "metadata": {"source": "faq.txt"},
        },
    ]
    
    keywords = ["password", "reset"]
    ranked = hybrid_rank(vector_results, keywords, vector_weight=0.8, keyword_weight=0.2)
    
    # First result should still rank high (high vector + high keywords)
    assert len(ranked) == 2
    assert ranked[0]["text"] == "Password reset steps for account access"
    assert "keyword_score" in ranked[0]
    assert "hybrid_score" in ranked[0]
    assert ranked[0]["keyword_score"] == 2.0  # "password" and "reset" both present


def test_hybrid_rank_reorders_by_keywords():
    """Test that hybrid_rank can reorder results based on keyword matches."""
    vector_results = [
        {
            "score": 0.95,  # Higher vector score
            "text": "Generic security information",
            "metadata": {"source": "guide.txt"},
        },
        {
            "score": 0.6,   # Lower vector score
            "text": "Password reset password change password recovery",
            "metadata": {"source": "faq.txt"},
        },
    ]
    
    keywords = ["password"]
    # Use higher keyword weight to favor exact matches
    ranked = hybrid_rank(vector_results, keywords, vector_weight=0.5, keyword_weight=0.5)
    
    # Second result should rank higher due to multiple keyword matches
    assert ranked[0]["text"] == "Password reset password change password recovery"


def test_hybrid_rank_invalid_weights():
    """Test that hybrid_rank raises error for invalid weights."""
    results = [{"score": 0.9, "text": "Test text", "metadata": {}}]
    keywords = ["test"]
    
    with pytest.raises(ValueError, match="weights must sum"):
        hybrid_rank(results, keywords, vector_weight=0.6, keyword_weight=0.2)


def test_retrieve_with_embedding_passes_filter():
    """Test that retrieve_with_embedding passes metadata filter to retrieve."""
    call_log = []
    
    def mock_query(self, **kwargs):
        call_log.append(kwargs)
        return {
            "ids": [["doc-1:0"]],
            "distances": [[0.05]],
            "documents": [["Filtered chunk"]],
            "metadatas": [[{"source": "doc.txt", "chunk_index": 0}]],
        }
    
    mock_collection = type("Collection", (), {"query": mock_query})()
    
    def mock_embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    metadata_filter = {"section": "Important"}
    retrieve_with_embedding(
        mock_collection,
        "test query",
        mock_embed,
        k=3,
        metadata_filter=metadata_filter
    )
    
    # Verify filter was passed through
    assert len(call_log) == 1
    assert call_log[0].get("where") == metadata_filter

