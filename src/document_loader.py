from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup


def load_text(path: Path) -> str:
    """Load a supported document and return plain text."""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if suffix in (".txt", ".md"):
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    if suffix in (".html", ".htm"):
        html = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
        return BeautifulSoup(
            html,
            "html.parser"
        ).get_text(" ", strip=True)

    raise ValueError(f"unsupported format: {suffix}")


def load_documents(data_dir: str = "data"):
    """Load all documents while skipping files that fail."""

    documents = []

    for path in Path(data_dir).rglob("*"):

        # Ignore directories and Git placeholder files
        if not path.is_file() or path.name == ".gitkeep":
            continue

        try:
            text = load_text(path)

            documents.append({
                "source": path.name,
                "text": text
            })

            print(
                f"OK {path.name}: "
                f"{len(text)} chars | "
                f"{text[:60]!r}"
            )

        except Exception as e:
            print(f"SKIP {path.name}: {e}")

    return documents


if __name__ == "__main__":
    load_documents()