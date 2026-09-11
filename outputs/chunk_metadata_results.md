# Chunk Metadata Results

The chunkers now return text together with metadata. These records came from
the paragraph chunker running on `data/sample.txt`:

```python
{
    "text": "Contexto helps journalists retrieve accurate historical information. This document contains research notes.",
    "metadata": {
        "source": "sample.txt",
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 107,
        "section": None,
        "page": None,
    },
}
```

```python
{
    "text": "The journalist's archive contains historical research and interview notes.",
    "metadata": {
        "source": "sample.txt",
        "chunk_index": 1,
        "char_start": 109,
        "char_end": 183,
        "section": None,
        "page": None,
    },
}
```

## Trace Example

For a retrieved item, `hit["metadata"]` identifies the document and exact
character range needed to cite it:

```python
hit = retrieved_chunks[1]
metadata = hit["metadata"]
print(
    f"Answer from {metadata['source']} "
    f"(chunk {metadata['chunk_index']}, "
    f"chars {metadata['char_start']}-{metadata['char_end']})"
)
```

```text
Answer from sample.txt (chunk 1, chars 109-183)
```