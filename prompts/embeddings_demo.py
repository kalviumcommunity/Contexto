"""Generate OpenAI embeddings for sample texts and compare similarity."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a list of text strings."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_vec = np.asarray(a, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    return float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))


def main() -> None:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY was not found in .env")

    if "your_" in API_KEY.lower() or "replace_me" in API_KEY.lower() or "example" in API_KEY.lower():
        raise RuntimeError(
            "OPENAI_API_KEY is still set to the placeholder value in .env. "
            "Replace it with a valid OpenAI API key before running the embeddings demo."
        )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    texts = [
        "How do I reset my account password?",
        "Steps to recover access to my login",
        "The cafeteria menu has pasta today",
    ]

    try:
        embeddings = embed_texts(client, texts)
    except AuthenticationError as exc:
        raise RuntimeError(
            "Authentication failed. Check that OPENAI_API_KEY is valid and matches the correct OpenAI project."
        ) from exc

    print("dimension:", len(embeddings[0]))
    print("first 8 values:", embeddings[0][:8])

    password_vs_login = cosine_similarity(embeddings[0], embeddings[1])
    password_vs_menu = cosine_similarity(embeddings[0], embeddings[2])

    print("password vs login recovery:", password_vs_login)
    print("password vs cafeteria menu:", password_vs_menu)

    print("\nInterpretation:")
    print(
        "The password and login-recovery texts should score higher because they express "
        "similar intent, while the cafeteria-menu text should be farther away in vector space."
    )


if __name__ == "__main__":
    main()
