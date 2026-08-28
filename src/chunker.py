from document_loader import load_documents
from text_cleaner import clean_text


def fixed_chunks(text: str, size: int = 500, overlap: int = 50):
    """Split text into fixed-size chunks with overlap."""

    if size <= overlap:
        raise ValueError("chunk size must be greater than overlap")

    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        chunk = text[start:start + size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def paragraph_chunks(text: str):
    """Split text using paragraph boundaries."""

    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


def report_chunks(name: str, chunks: list[str]):
    """Print chunk statistics."""

    if not chunks:
        print(f"{name}: 0 chunks")
        return

    sizes = [len(chunk) for chunk in chunks]
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

        fixed = fixed_chunks(text, size=150, overlap=30)
        paragraph = paragraph_chunks(text)

        report_chunks("Fixed-size", fixed)
        report_chunks("Paragraph", paragraph)

        if fixed:
            print("\nFixed-size sample:")
            print(fixed[0][:200])

        if paragraph:
            print("\nParagraph sample:")
            print(paragraph[0][:200])