from prompts.vector_indexing import batches, to_vector_record


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
