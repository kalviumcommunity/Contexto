import re
import unicodedata

from document_loader import load_documents


def fix_encoding_artifacts(text: str) -> str:
    """Fix common UTF-8/Windows-1252 encoding artifacts."""

    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â ": " ",
        "Â": "",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Handle the terminal representation of common mojibake.
    text = text.replace("â€TM", "'")
    text = text.replace("â€TM", "'")

    return text


def clean_text(text: str) -> str:
    """Clean extracted text for consistent downstream processing."""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Fix encoding artifacts.
    text = fix_encoding_artifacts(text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page-number boilerplate.
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove repeated archive headers.
    text = re.sub(
        r"(?mi)^\s*CONTEXTO MEDIA ARCHIVE\s*$",
        "",
        text
    )

    # Collapse repeated spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(data_dir: str = "data"):
    """Load and clean every supported document consistently."""

    documents = load_documents(data_dir)

    for document in documents:
        before = document["text"]
        document["text"] = clean_text(before)

        print(
            f"{document['source']}: "
            f"{len(before)} -> {len(document['text'])} chars"
        )

    return documents


if __name__ == "__main__":
    documents = load_documents()

    if documents:
        # Select the noisy sample document for demonstration.
        example = next(
            (
                document
                for document in documents
                if "CONTEXTO MEDIA ARCHIVE" in document["text"]
            ),
            documents[0]
        )

        example_before = example["text"]
        example_after = clean_text(example_before)

        print("\nBEFORE:")
        print(example_before[:300])

        print("\nAFTER:")
        print(example_after[:300])

    print("\nCleaning all documents:")

    for document in documents:
        before = document["text"]
        document["text"] = clean_text(before)

        print(
            f"{document['source']}: "
            f"{len(before)} -> {len(document['text'])} chars"
        )