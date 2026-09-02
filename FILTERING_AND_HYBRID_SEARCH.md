# Metadata Filtering & Hybrid Search Implementation

## Overview

This implementation adds **Metadata Filtering** and **Hybrid Search** capabilities to the Contexto RAG system, enabling more precise and relevant retrieval by:

1. **Scoping searches** with metadata filters (source, section, document type, etc.)
2. **Combining semantic and lexical matching** through hybrid ranking
3. **Tuning weights** based on use case priorities

## Architecture

### Retrieval Pipeline with Filtering

```
User Query
    ↓
[Embed Query]
    ↓
[Optional Metadata Filter]
    ↓
[Vector Search (Semantic)]
    ↓
[Optional Hybrid Ranking (Semantic + Keyword)]
    ↓
Ranked, Filtered Results
```

## API Reference

### Enhanced Functions

#### `retrieve(collection, query_vector, *, k=3, metadata_filter=None)`

Enhanced vector search with optional metadata filtering.

**Parameters:**
```python
collection          # Chroma collection
query_vector        # Pre-computed embedding [float, ...]
k                   # Number of results (default: 3)
metadata_filter     # Optional: {"field": "value"} or Chroma filter dict
```

**Returns:**
```python
[
    {
        "score": 0.95,
        "text": "chunk content...",
        "metadata": {
            "source": "doc.txt",
            "chunk_index": 0,
            "section": "Intro"
        }
    },
    ...
]
```

**Example:**
```python
# Unfiltered search
results = retrieve(collection, query_embedding, k=5)

# Filtered to specific section
results = retrieve(
    collection,
    query_embedding,
    k=5,
    metadata_filter={"section": "Account access"}
)

# Multiple filter conditions (Chroma syntax)
results = retrieve(
    collection,
    query_embedding,
    k=5,
    metadata_filter={
        "$and": [
            {"section": "Account access"},
            {"source": {"$eq": "faq.txt"}}
        ]
    }
)
```

#### `retrieve_with_embedding(collection, query, embed_fn, *, k=3, metadata_filter=None)`

High-level function combining embedding and filtered retrieval.

**Parameters:**
```python
collection          # Chroma collection
query              # Raw query string from user
embed_fn           # Embedding function: str[] → float[][]
k                  # Number of results (default: 3)
metadata_filter    # Optional metadata filter dict
```

**Example:**
```python
results = retrieve_with_embedding(
    collection=collection,
    query="How do I reset my password?",
    embed_fn=lambda texts: embed_texts(openai_client, texts),
    k=3,
    metadata_filter={"section": "Account access"}
)
```

### New Hybrid Search Functions

#### `keyword_score(text, keywords)`

Counts keyword occurrences in text (lexical matching).

**Parameters:**
```python
text        # str - Text to search
keywords    # Sequence[str] - Keywords to match
```

**Returns:**
```python
float       # Count of keyword matches (case-insensitive)
```

**Example:**
```python
text = "Password reset password verification password change"
keywords = ["password", "reset"]
score = keyword_score(text, keywords)  # → 4.0
```

#### `hybrid_rank(vector_results, keywords, *, vector_weight=0.8, keyword_weight=0.2)`

Combines vector similarity and keyword matching for hybrid ranking.

**Parameters:**
```python
vector_results      # list[dict] - Results from vector search
keywords           # Sequence[str] - Keywords to match
vector_weight      # float - Weight for vector score (default: 0.8)
keyword_weight     # float - Weight for keyword score (default: 0.2)
```

**Returns:**
```python
[
    {
        ...vector_result_fields...,
        "keyword_score": 2.0,        # Number of keyword matches
        "hybrid_score": 0.78,        # Weighted combination
    },
    ...  # Sorted by hybrid_score descending
]
```

**Example:**
```python
# Basic hybrid ranking (80% vector, 20% keyword)
hybrid = hybrid_rank(vector_results, ["password", "reset"])

# Keyword-heavy for product name searches
hybrid = hybrid_rank(
    vector_results,
    ["SKU-12345"],
    vector_weight=0.5,
    keyword_weight=0.5
)

# Semantic-focused (find related concepts)
hybrid = hybrid_rank(
    vector_results,
    ["password"],
    vector_weight=0.95,
    keyword_weight=0.05
)
```

## Use Cases & Strategies

### Use Case 1: Scoped Retrieval by Section

**Goal:** Help users find content in specific areas

```python
# User asks: "Show me account access help"
results = retrieve_with_embedding(
    collection,
    "How do I reset my password?",
    embed_fn,
    k=3,
    metadata_filter={"section": "Account access"}
)
```

**Benefits:**
- ✓ Precise results from user's intended section
- ✓ Avoids distracting content from other sections
- ✓ Lower cost (fewer irrelevant results)

### Use Case 2: Product-Specific Search

**Goal:** Retrieve only from approved public documentation

```python
results = retrieve_with_embedding(
    collection,
    "payment processing",
    embed_fn,
    k=5,
    metadata_filter={
        "$and": [
            {"product": "payment-gateway"},
            {"access_level": "public"}
        ]
    }
)
```

**Benefits:**
- ✓ Ensures compliance (public docs only)
- ✓ Reduces hallucination (no internal-only content)
- ✓ Improves trust (verifiable sources)

### Use Case 3: Exact Match Boosting

**Goal:** Find results mentioning exact product codes

```python
query = "I need help with SKU-98765-A"
vector_results = retrieve(collection, query_embedding, k=10)

# Boost results that mention the exact code
hybrid = hybrid_rank(
    vector_results,
    ["SKU-98765-A"],
    vector_weight=0.5,
    keyword_weight=0.5
)
```

**Benefits:**
- ✓ Pure semantic search might find related SKUs
- ✓ Hybrid ensures exact match gets top ranking
- ✓ Good for IDs, codes, proper names

### Use Case 4: Combined Filtering + Hybrid

**Goal:** Balance precision and recall for complex queries

```python
# User: "How do I reset my password for the mobile app?"
query = "reset password"
keywords = ["reset", "password", "mobile", "app"]

# Step 1: Filter to relevant sections
filtered = retrieve(
    collection,
    query_embedding,
    k=10,
    metadata_filter={
        "$or": [
            {"section": "Account access"},
            {"section": "Mobile app"}
        ]
    }
)

# Step 2: Hybrid rank within filtered results
results = hybrid_rank(
    filtered,
    keywords,
    vector_weight=0.7,
    keyword_weight=0.3
)
```

**Benefits:**
- ✓ Filtering reduces search space (faster, cheaper)
- ✓ Hybrid ranking finds best match within scope
- ✓ Balances precision + semantic understanding

## Vector vs. Keyword Search Trade-Offs

### Vector Search (Semantic)
```
Input:  "How do I reset my password?"
Match:  "To change your access credentials..."
        (different wording, same meaning)
```

**Pros:**
- ✓ Finds related concepts even with different wording
- ✓ Handles synonyms and paraphrasing
- ✓ Better for natural language queries
- ✓ Faster (single query to vector DB)

**Cons:**
- ✗ May miss exact-match requirements
- ✗ Can't distinguish SKU-123 from SKU-456
- ✗ Struggles with proper names and acronyms

### Keyword Search (Lexical)
```
Input:  "product code"
Match:  "PROD-2024-Q1" appears exactly in text
```

**Pros:**
- ✓ Finds exact matches
- ✓ Great for codes, IDs, proper names
- ✓ Predictable behavior

**Cons:**
- ✗ Misses paraphrased or similar content
- ✗ Fails if wording varies slightly
- ✗ Less effective for natural language

### Hybrid Search
Combines both approaches:
- ✓ Semantic understanding + exact match reliability
- ✓ Flexible weighting for different use cases
- ✓ Better coverage than either alone

## Metadata Schema Design

Good metadata supports effective filtering:

```python
metadata = {
    "source": "help_article.txt",        # Document origin
    "chunk_index": 5,                    # Position in document
    "section": "Account Management",     # Logical section
    "subsection": "Password",            # Nested category
    "product": "web-dashboard",          # Product area
    "access_level": "public",            # Audience scope
    "document_type": "faq",              # Content type
    "language": "en",                    # Language
    "created_date": "2024-01-15",        # Temporal info
    "category": "troubleshooting",       # Content category
}
```

**Filtering Examples:**

```python
# By section
{"section": "Account Management"}

# By product and access level
{"$and": [
    {"product": "web-dashboard"},
    {"access_level": "public"}
]}

# Multiple options
{"$or": [
    {"document_type": "faq"},
    {"document_type": "guide"}
]}

# Combining AND + OR
{"$and": [
    {"access_level": "public"},
    {"$or": [
        {"product": "web-dashboard"},
        {"product": "mobile-app"}
    ]}
]}
```

## Implementation Details

### Keyword Scoring Algorithm

```python
def keyword_score(text, keywords):
    """Count keyword occurrences (case-insensitive)."""
    lowered = text.lower()
    return sum(keyword.lower() in lowered for keyword in keywords)
```

- Simple substring matching (not word boundaries)
- Case-insensitive comparison
- Counts each occurrence (not unique keywords)

### Hybrid Scoring Formula

```python
hybrid_score = (vector_weight * vector_score) + (keyword_weight * normalized_lexical_score)
```

Where:
- `vector_score`: 0-1 (from vector search)
- `lexical_score`: number of keyword matches
- `normalized_lexical_score`: min(lexical_score / len(keywords), 1.0)
- `vector_weight + keyword_weight ≈ 1.0`

### Weight Selection Guide

| Use Case | Vector | Keyword | Reasoning |
|----------|--------|---------|-----------|
| General Q&A | 0.9 | 0.1 | Semantic understanding primary |
| Product codes | 0.5 | 0.5 | Balance exact + related matches |
| Names/proper nouns | 0.4 | 0.6 | Exact names are critical |
| Troubleshooting | 0.8 | 0.2 | Related problems matter most |
| Policy lookup | 0.3 | 0.7 | Exact policy text important |

## Testing

16 comprehensive tests cover:
- ✅ Vector retrieval without filtering
- ✅ Metadata filtering behavior
- ✅ Keyword scoring (matching, case-insensitivity, edge cases)
- ✅ Hybrid ranking (score combination, reordering, weight validation)
- ✅ Integration (filters passed through to retrieve_with_embedding)

**Run tests:**
```bash
pytest tests/test_vector_indexing.py -v
```

## Files Modified/Created

- **Modified**: `prompts/vector_indexing.py` - Added filtering and hybrid functions
- **Modified**: `tests/test_vector_indexing.py` - Added 9 new tests (16 total)
- **Created**: `prompts/filtering_demo.py` - Interactive demonstration

## Demo Script

Run the comprehensive filtering + hybrid search demo:

```bash
python prompts/filtering_demo.py
```

Shows:
1. Unfiltered vs. filtered retrieval comparison
2. Vector vs. keyword search differences
3. Hybrid ranking in action
4. Combined filtering + hybrid strategy
5. Real-world use case recommendations

## Next Steps in RAG Pipeline

Current capabilities:
1. ✅ Document chunking & embedding
2. ✅ Vector database indexing
3. ✅ Top-k retrieval
4. ✅ **Metadata filtering & hybrid search** ← YOU ARE HERE
5. Grounding answer generation with retrieved context
6. Citation and source attribution

The retrieved and ranked chunks are now ready to format with source attribution and pass to the language model for grounded answer generation.

## References

- [Chroma Filtering Docs](https://docs.trychroma.com/reference/client#query-chroma)
- [Hybrid Search Patterns](https://weaviate.io/developers/weaviate/search/hybrid)
- [Metadata Best Practices](https://docs.pinecone.io/guides/organize-data/understanding-filtering)
