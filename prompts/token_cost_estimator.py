"""Count project text tokens and estimate separate input/output costs."""

from dataclasses import dataclass
from pathlib import Path

import tiktoken


INPUT_PRICE_PER_1K = 0.0005
OUTPUT_PRICE_PER_1K = 0.0015
ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class TextSample:
    name: str
    input_text: str
    output_text: str


def count_tokens(encoder: tiktoken.Encoding, text: str) -> int:
    """Return the number of tokens produced by the selected encoding."""
    return len(encoder.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate one request cost using separate per-1K input/output rates."""
    return (
        input_tokens / 1000 * INPUT_PRICE_PER_1K
        + output_tokens / 1000 * OUTPUT_PRICE_PER_1K
    )


def build_samples(project_root: Path) -> list[TextSample]:
    """Build short, paragraph, and full-document samples from project text."""
    return [
        TextSample(
            name="short question",
            input_text="What is our refund window?",
            output_text="I don't know based on the available information.",
        ),
        TextSample(
            name="research paragraph",
            input_text=(
                "Contexto helps journalists retrieve historical context from articles, "
                "interview transcripts, and archived footage notes. Answers should stay "
                "grounded in the retrieved sources."
            ),
            output_text=(
                "Contexto retrieves source-backed historical context for journalists "
                "and should avoid unsupported claims."
            ),
        ),
        TextSample(
            name="full README document",
            input_text=(project_root / "README.md").read_text(encoding="utf-8"),
            output_text=(
                "The project is a RAG assistant for retrieving accurate historical "
                "context and showing the sources behind an answer."
            ),
        ),
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    encoder = tiktoken.get_encoding(ENCODING_NAME)
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"Tokenizer: {ENCODING_NAME}")
    print(
        f"Rates: input=${INPUT_PRICE_PER_1K:.4f}/1K, "
        f"output=${OUTPUT_PRICE_PER_1K:.4f}/1K"
    )
    print()

    for sample in build_samples(project_root):
        input_tokens = count_tokens(encoder, sample.input_text)
        output_tokens = count_tokens(encoder, sample.output_text)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        input_chars = len(sample.input_text)
        output_chars = len(sample.output_text)
        print(sample.name)
        print(
            f"  input:  {input_chars} chars, {len(sample.input_text.split())} words, "
            f"{input_tokens} tokens"
        )
        print(
            f"  output: {output_chars} chars, {len(sample.output_text.split())} words, "
            f"{output_tokens} tokens"
        )
        print(f"  estimated cost: ${estimate_cost(input_tokens, output_tokens):.6f}")
        print()

    print(f"Total estimated cost: ${estimate_cost(total_input_tokens, total_output_tokens):.6f}")
    print("Token counts increase with text length, but characters and tokens are not proportional.")


if __name__ == "__main__":
    main()