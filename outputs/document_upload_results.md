# Document Upload & Indexing Results

## Upload Endpoint

Endpoint: POST /documents

Supported formats: .txt, .md, .pdf

## Processing Flow

Upload -> Validate -> Store -> Load -> Clean -> Chunk -> Embed -> Index

## Sample Upload Request

curl -X POST http://127.0.0.1:8000/documents -F "file=@data/new-policy.md"

## Sample Document

new-policy.md contains information about requesting project extensions from the academic support team.

## Expected Indexing Summary

status: indexed
filename: new-policy.md
document: uploads/new-policy.md
chunks: 1
indexed: 1

## Runtime Searchability

The upload pipeline inserts new records into the existing rag_chunks Chroma collection. Subsequent queries can therefore access newly indexed content without restarting the application.

## Error Handling

Unsupported file type: HTTP 415

Empty file: HTTP 400

File larger than 5 MB: HTTP 413

Unexpected processing/indexing error: HTTP 500

## Pipeline Reuse

The endpoint reuses the existing document loader, text cleaner, token chunker, embedding interface, and Chroma vector-store components.

## Offline Embeddings

The implementation uses deterministic demo vectors for local/offline development because the available embedding API credits are exhausted.

## Runtime Test Note

The live API test was attempted, but the local environment does not currently have the FastAPI package installed. The endpoint source passed Python syntax validation.

## Large Documents

For very large uploads, the recommended approach is to store the file first, create a background processing job, process chunks in batches, report progress, and retry transient failures.
