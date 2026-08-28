from document_loader import load_documents
from text_cleaner import clean_text


def tag_chunks(
    source: str,
    chunks: list[tuple[str, int]],
    *,
    section: str | None = None,
    page: int | None = None,
) -> list[dict]:
    """Attach a consistent citation record to every chunk."""

    tagged = []
    for index, (text, char_start) in enumerate(chunks):
        tagged.append(
            {
                "text": text,
                "metadata": {
                    "source": source,
                    "chunk_index": index,
                    "char_start": char_start,
                    "char_end": char_start + len(text),
                    "section": section,
                    "page": page,
                },
            }
        )
    return tagged


def _trimmed_chunk(text: str, start: int, end: int) -> tuple[str, int]:
    raw_chunk = text[start:end]
    chunk = raw_chunk.strip()
    if not chunk:
        return "", start

    leading_whitespace = len(raw_chunk) - len(raw_chunk.lstrip())
    return chunk, start + leading_whitespace


def fixed_chunks(
    text: str,
    source: str,
    size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split text into fixed-size chunks with overlap."""

    if size <= overlap:
        raise ValueError("chunk size must be greater than overlap")

    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        chunk, char_start = _trimmed_chunk(text, start, start + size)

        if chunk:
            chunks.append((chunk, char_start))

        start += step

    return tag_chunks(source, chunks)


def paragraph_chunks(text: str, source: str) -> list[dict]:
    """Split text using paragraph boundaries."""

    chunks = []
    search_start = 0
    for paragraph in text.split("\n\n"):
        paragraph_start = text.find(paragraph, search_start)
        chunk, char_start = _trimmed_chunk(
            text, paragraph_start, paragraph_start + len(paragraph)
        )
        if chunk:
            chunks.append((chunk, char_start))
        search_start = paragraph_start + len(paragraph) + 2

    return tag_chunks(source, chunks)


def report_chunks(name: str, chunks: list[dict]):
    """Print chunk statistics."""

    if not chunks:
        print(f"{name}: 0 chunks")
        return

    sizes = [len(chunk["text"]) for chunk in chunks]
    average_size = sum(sizes) // len(sizes)

    print(
        f"{name}: "
        f"{len(chunks)} chunks, "
        f"avg {average_size} chars"
    )


if __name__ == "__main__":
    documents = load_documents()

    for document in documents:
        text = clean_text(document["text"])

        print("\n" + "=" * 60)
        print(f"SOURCE: {document['source']}")
        print("=" * 60)

        fixed = fixed_chunks(
            text, source=document["source"], size=150, overlap=30
        )
        paragraph = paragraph_chunks(text, source=document["source"])

        report_chunks("Fixed-size", fixed)
        report_chunks("Paragraph", paragraph)

        if fixed:
            print("\nFixed-size sample:")
            print(fixed[0])

        if paragraph:
            print("\nParagraph sample:")
            print(paragraph[0])

        if fixed:
            metadata = fixed[0]["metadata"]
            print(
                "Trace: "
                f"{metadata['source']} "
                f"(chunk {metadata['chunk_index']}, "
                f"chars {metadata['char_start']}-{metadata['char_end']})"
            )