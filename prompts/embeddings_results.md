# Embeddings Fundamentals and Vector Similarity

This example creates embeddings for short texts and measures whether semantically similar phrases appear closer in vector space than unrelated ones.

Run it with:

```bash
python prompts/embeddings_demo.py
```

## Vector basics

An embedding is a list of numeric coordinates that represents the meaning of a text. The vector length is the embedding dimension.

```python
embeddings = embed(texts)
print("dimension:", len(embeddings[0]))
print("first 8 values:", embeddings[0][:8])
```

If the model produces 1536-dimensional vectors, then each text is mapped to 1536 numeric values. You rarely interpret one coordinate directly; the complete pattern is what captures meaning.

## Similarity

Cosine similarity compares the direction of two vectors rather than their raw magnitude.

```python
from numpy import dot
from numpy.linalg import norm


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))
```

A pair like "reset my account password" and "recover access to my login" should score higher than the pair with a cafeteria menu example because they express similar intent even though the words differ.

## Why this matters for RAG

In a production RAG pipeline, every chunk is embedded and stored in a vector database. A user question is embedded the same way, and retrieval becomes nearest-neighbor search: the system finds the chunks whose vectors are closest to the question vector, even when the wording differs.

## Interpretation

Embeddings let retrieval work semantically instead of only by exact keyword match. This is what allows a question about "password recovery" to match a chunk that says "account access reset steps," even though the words are not identical.
