 Document-Loading
"""Load the Contexto sample corpus into plain-text documents."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

from bs4 import BeautifulSoup
from pypdf import PdfReader


class LoadedDocument(TypedDict):
    source: str
    text: str


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}


def load_text(path: Path) -> str:
    """Extract plain text from one supported document."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return "\n".join(
            page.extract_text() or "" for page in PdfReader(str(path)).pages
        )
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

    raise ValueError(f"unsupported file type: {suffix or '[no extension]'}")


def load_corpus(data_dir: Path) -> list[LoadedDocument]:
    """Load supported files below ``data_dir``, skipping files that fail."""
    documents: list[LoadedDocument] = []

    if not data_dir.exists():
        print(f"SKIP {data_dir}: directory does not exist")
        return documents
    if not data_dir.is_dir():
        print(f"SKIP {data_dir}: expected a directory")
        return documents

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = load_text(path)
        except Exception as error:
            print(f"SKIP {path.name}: {error}")
            continue

        documents.append({"source": path.name, "text": text})
        print(f"OK {path.name}: {len(text)} chars | {text[:60]!r}")

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Contexto document corpus")
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="directory containing documents (default: ./data)",
    )
    args = parser.parse_args()
    documents = load_corpus(args.data_dir)
    print(f"Loaded {len(documents)} document(s).")


if __name__ == "__main__":
    main()
=======
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
 main
