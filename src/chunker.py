from document_loader import load_documents
from text_cleaner import clean_text


TOKENIZER_NAME = "cl100k_base"


def tag_chunks(
    source: str,
    chunks: list[tuple[str, int]],
    *,
    section: str | None = None,
    page: int | None = None,
    token_ranges: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Attach a consistent citation record to every chunk."""

    tagged = []
    for index, (text, char_start) in enumerate(chunks):
        token_start, token_end = (None, None)
        if token_ranges is not None:
            token_start, token_end = token_ranges[index]
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
                    "token_start": token_start,
                    "token_end": token_end,
                    "token_count": (
                        token_end - token_start
                        if token_start is not None and token_end is not None
                        else None
                    ),
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


def token_chunks(
    text: str,
    source: str,
    size: int = 400,
    overlap: int = 60,
) -> list[dict]:
    """Split text into token-sized chunks with controlled overlap."""

    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and less than chunk size")

    import tiktoken

    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    tokens = encoder.encode(text)
    chunks = []
    token_ranges = []
    start = 0
    step = size - overlap

    while start < len(tokens):
        end = min(start + size, len(tokens))
        overlap_start = max(0, start - overlap)
        chunk_tokens = tokens[overlap_start:end]
        chunk = encoder.decode(chunk_tokens)
        decoded_prefix = encoder.decode(tokens[:start])
        char_start = text.find(chunk, len(decoded_prefix))
        if char_start < 0:
            char_start = len(decoded_prefix)
        chunks.append((chunk, char_start))
        token_ranges.append((start, end))
        start += step

    return tag_chunks(source, chunks, token_ranges=token_ranges)


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