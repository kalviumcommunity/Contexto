# Similarity Search & Top-K Retrieval Implementation

## Overview

This implementation completes the **Similarity Search & Top-K Retrieval** stage of the Contexto RAG system, enabling the application to:

1. **Embed queries** using the same OpenAI embedding model as indexed documents
2. **Search the vector database** (Chroma) for semantically similar chunks
3. **Return ranked results** with similarity scores and metadata for citation and verification
4. **Tune retrieval quality** by adjusting the k parameter

## Implementation

### Core Functions in `prompts/vector_indexing.py`

#### `retrieve(collection, query_vector, *, k=3)`
Performs similarity search against the vector database.

**Parameters:**
- `collection`: Chroma collection containing indexed chunks
- `query_vector`: The embedding of the user's query (pre-embedded)
- `k`: Number of top results to return (default: 3)

**Returns:**
A list of result dictionaries, each containing:
```python
{
    "score": 0.95,              # Similarity score (0-1, higher = more similar)
    "text": "chunk content...",  # The actual chunk text
    "metadata": {               # Source information for attribution
        "source": "article.txt",
        "chunk_index": 0,
        "section": "Introduction"
    }
}
```

#### `retrieve_with_embedding(collection, query, embed_fn, *, k=3)`
High-level retrieval function that combines embedding and search.

**Parameters:**
- `collection`: Chroma collection
- `query`: Raw query string from the user
- `embed_fn`: Embedding function (e.g., `lambda texts: embed_texts(client, texts)`)
- `k`: Number of results to return

**Usage:**
```python
# Embed and retrieve in one call
results = retrieve_with_embedding(
    collection=collection,
    query="How do I reset my password?",
    embed_fn=lambda texts: embed_texts(openai_client, texts),
    k=3
)
```

### Helper Functions

#### `_normalize_search_result(result)`
Normalizes Chroma's response format into a consistent structure. Handles both dict-based and tuple-based formats across different Chroma versions.

### Key Design Decisions

1. **Separate embedding from search**: `retrieve()` takes pre-embedded vectors, allowing flexibility in embedding strategies and reducing redundant API calls.

2. **Optional high-level wrapper**: `retrieve_with_embedding()` handles the full pipeline for convenience.

3. **Format normalization**: The `_normalize_search_result()` function ensures compatibility with different Chroma versions.

4. **Error handling**: Invalid k values are caught early; collection query failures fall back to alternate APIs.

## Testing

Seven comprehensive tests cover:
- ✅ Vector record conversion
- ✅ Batch processing
- ✅ Search result normalization (dict and tuple formats)
- ✅ Input validation (k must be > 0)
- ✅ Chroma response handling
- ✅ k parameter propagation

**Run tests:**
```bash
pytest tests/test_vector_indexing.py -v
```

## Top-K Retrieval Trade-Offs

### Small k (1-3)
- ✅ **Pros**: Fast, low cost, minimal context overhead
- ❌ **Cons**: May miss relevant context; incomplete answers possible

### Large k (5-10)
- ✅ **Pros**: Better recall; more grounding material for LLM
- ❌ **Cons**: Slower, higher cost, risk of including noise; larger prompt context

### Tuning Strategy
Adjust k based on:
1. **Chunk size**: Smaller chunks → higher k
2. **Question complexity**: Complex questions → higher k
3. **Context window**: Consider your LLM's limits
4. **Quality vs. speed trade-off**: Start with k=3, measure retrieval quality

## Usage Examples

### Example 1: Simple Retrieval
```python
import chromadb
from prompts.embeddings_demo import embed_texts
from prompts.vector_indexing import retrieve

client = chromadb.PersistentClient(path="./.chroma")
collection = client.get_or_create_collection("contexto_chunks")

# Get query embedding
query_embedding = embed_texts(openai_client, ["What is this article about?"])[0]

# Retrieve top 3 chunks
results = retrieve(collection, query_embedding, k=3)

for rank, result in enumerate(results, start=1):
    print(f"Rank {rank}: {result['score']:.4f}")
    print(f"Source: {result['metadata']['source']}")
    print(f"Text: {result['text'][:100]}...")
```

### Example 2: Full Pipeline
```python
results = retrieve_with_embedding(
    collection=collection,
    query="How can a learner reset their password?",
    embed_fn=lambda texts: embed_texts(openai_client, texts),
    k=5
)

# Results are ready for passing to LLM
for result in results:
    print(result["text"])
    print(result["metadata"])
```

## Integration with RAG Pipeline

The retrieval output feeds directly into the grounded answer generation:

```
User Query
    ↓
[Embed Query] (same model as document chunks)
    ↓
[Search Vector DB] (cosine similarity)
    ↓
[Retrieve Top-K Chunks] ← YOU ARE HERE
    ↓
[Format as Context]
    ↓
[LLM with Grounded Prompt]
    ↓
Citation-Ready Answer
```

## Files Modified/Created

- **Modified**: `prompts/vector_indexing.py` - Added retrieval functions
- **Modified**: `tests/test_vector_indexing.py` - Added comprehensive tests
- **Created**: `prompts/retrieval_demo.py` - Demonstration script

## Next Steps

With retrieval working, the RAG pipeline can now:
1. ✅ Ingest and chunk documents
2. ✅ Generate and store embeddings
3. ✅ Index chunks in vector database
4. ✅ **Retrieve top-k similar chunks** ← COMPLETED
5. Format retrieved context with source attribution
6. Pass grounded context to LLM for answer generation

The next assignment will likely cover **Metadata Filtering & Hybrid Search**, enabling more sophisticated retrieval strategies (e.g., filtering by source, combining keyword and semantic search).
