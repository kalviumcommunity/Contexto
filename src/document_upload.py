from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from document_loader import load_documents
from text_cleaner import clean_text
from chunker import token_chunks
from embeddings import demo_vectors
from vector_store import (
    create_client,
    create_collection,
    build_record,
    upsert_record,
)


UPLOAD_DIR = Path("uploads")

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}

MAX_FILE_SIZE = 5 * 1024 * 1024


def validate_upload(file: UploadFile) -> str:
    """Validate filename and supported extension."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Use .txt, .md, or .pdf",
        )

    return suffix


async def store_upload(file: UploadFile) -> Path:
    """Store an uploaded file safely inside the uploads directory."""

    validate_upload(file)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    path = UPLOAD_DIR / safe_name

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the 5 MB size limit",
        )

    path.write_bytes(content)

    return path


def process_uploaded_document(path: Path) -> dict[str, Any]:
    """Load, clean, chunk, embed and index an uploaded document."""

    documents = load_documents(path.parent)

    document = next(
        (
            item
            for item in documents
            if item["source"] == path.name
        ),
        None,
    )

    if document is None:
        raise ValueError(
            f"Unable to load uploaded document: {path.name}"
        )

    cleaned_text = clean_text(document["text"])

    if not cleaned_text:
        raise ValueError(
            f"Uploaded document contains no usable text: {path.name}"
        )

    chunks = token_chunks(
        cleaned_text,
        source=path.name,
        size=400,
        overlap=60,
    )

    if not chunks:
        raise ValueError(
            f"No chunks were produced for: {path.name}"
        )

    vectors = demo_vectors(chunks, dim=6)

    client = create_client()
    collection = create_collection(
        client,
        name="rag_chunks",
        vector_dimension=6,
        metric="cosine",
    )

    indexed = 0

    for index, (chunk, vector) in enumerate(
        zip(chunks, vectors)
    ):
        metadata = dict(chunk.get("metadata", {}))

        record_id = f"{path.name}:{index}"

        metadata["chunk_id"] = record_id

        record = build_record(
            record_id,
            vector,
            chunk["text"],
            metadata,
            vector_dimension=6,
        )

        upsert_record(collection, record)
        indexed += 1

    return {
        "document": str(path),
        "chunks": len(chunks),
        "indexed": indexed,
    }
