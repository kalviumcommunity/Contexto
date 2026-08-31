"""Split text into token-sized chunks with controlled overlap."""

import tiktoken

ENCODING_NAME = "cl100k_base"


def token_chunks(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """Return token-aware chunks with overlap between adjacent windows."""
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between 0 and size - 1")

    encoder = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoder.encode(text)
    chunks: list[str] = []
    index = 0

    while index < len(tokens):
        window = tokens[index : index + size]
        chunks.append(encoder.decode(window))
        if len(window) < size:
            break
        index += size - overlap

    return chunks


def sample_text() -> str:
    return (
        "Contexto is a newsroom research assistant built to help journalists find "
        "grounded answers from archived reports, interviews, and notes. A strong "
        "retrieval system keeps the answer faithful to the supplied sources instead "
        "of guessing from general knowledge. The chunking step matters because a "
        "document is often too large to fit into one retrieval window, and a hard "
        "cut can split a key fact in half. When text is split by tokens rather than "
        "characters, the model can work with a more predictable context budget. "
        "Overlapping the last 10 to 15 percent of each chunk preserves boundary "
        "meaning and gives the retriever a second chance to find the same fact in "
        "an adjacent fragment."
    )


def main() -> None:
    text = sample_text()
    encoder = tiktoken.get_encoding(ENCODING_NAME)

    print("Tokenizer: cl100k_base")
    for overlap in (0, 60):
        chunks = token_chunks(text, size=400, overlap=overlap)
        print(f"overlap {overlap}: {len(chunks)} chunks")

    print("\nBoundary preview:")
    chunks = token_chunks(text, size=80, overlap=15)
    for idx, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {idx} ({len(encoder.encode(chunk))} tokens)")
        print(chunk)

    print("\nWhy overlap helps:")
    print(
        "The same idea can appear in two adjacent chunks, so a retrieval step is less "
        "likely to lose a sentence or fact just because the boundary falls in the middle "
        "of a concept."
    )


if __name__ == "__main__":
    main()
