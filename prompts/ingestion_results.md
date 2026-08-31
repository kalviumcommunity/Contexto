# End-to-End Ingestion Validation

This example treats ingestion as one verified pipeline: load, clean, chunk, tag, and then confirm that every source file is accounted for.

Run it with:

```bash
python prompts/ingest_pipeline.py
```

## Full pipeline

```python
def ingest(folder):
    docs, chunks, failures = 0, [], []
    files = [p for p in Path(folder).rglob("*") if p.is_file()]
    for path in files:
        try:
            text = clean(load_text(path))
            tagged = tag_chunks(path.name, token_chunks(text))
            chunks += tagged
            docs += 1
        except Exception as e:
            failures.append((path.name, str(e)))
    return files, docs, chunks, failures
```

The key validation is this reconciliation:

```python
assert docs + len(failures) == len(files), "a document was silently dropped!"
```

That check prevents the common RAG failure mode where a few source documents fail silently and the assistant later cannot answer questions about them.

## Why this matters

When a retrieval system silently drops documents, there is no obvious model error and no clear upstream signal. The corpus looks complete in the directory listing but the answerable knowledge is incomplete. A pipeline summary makes those misses visible: total files, total ingested documents, total chunk objects, and explicit failures.

## Chunk metadata

Each output chunk includes:

- source file name
- chunk index
- chunk text
- metadata with the source, chunk index, and token count

This metadata matters during debugging because you can inspect a sample chunk and verify that the content and source attribution are correct before indexing or embedding it.

## Scaling to 4,000+ docs

For larger corpora, keep the same validation logic but add durability:

- log progress every N files
- write a manifest of processed files
- allow re-runs to skip already-finished items
- store a processed/failed list so a crash at file 3,900 does not restart from zero

The core invariant remains the same: `files == ingested + failures`.
