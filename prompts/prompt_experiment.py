"""Compare prompt designs for Contexto's staff-question assistant."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "You are Contexto, a support assistant for an internal media research team. "
    "Answer staff questions using only information provided in the conversation "
    "or retrieved context. Do not invent policies, dates, names, or sources. "
    "If the information is missing or uncertain, say: 'I don't know based on the "
    "available information.' Keep answers under 60 words, use a professional and "
    "direct tone, and include the relevant source when one is provided."
)

VARIATIONS = {
    "vague": "Explain our refund policy.",
    "constrained": (
        "Answer this staff question in one sentence: What is the refund window? "
        "Use only the supplied information and do not guess."
    ),
}


@dataclass(frozen=True)
class PromptResult:
    """The input and model output for one prompt variation."""

    variation: str
    user_prompt: str
    output: str


def build_messages(user_prompt: str) -> list[dict[str, str]]:
    """Keep assistant behavior in the system role and the question in user role."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def compare_prompts(client: "OpenAI", model: str) -> list[PromptResult]:
    """Run both prompt variations against the same model and question."""
    results = []
    for variation, user_prompt in VARIATIONS.items():
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(user_prompt),
        )
        results.append(
            PromptResult(
                variation=variation,
                user_prompt=user_prompt,
                output=response.choices[0].message.content or "",
            )
        )
    return results


def main() -> None:
    """Run the comparison when API configuration is available."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CHAT_MODEL")
    if not api_key or not model:
        raise SystemExit(
            "Set OPENAI_API_KEY and CHAT_MODEL in .env before running this experiment."
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install the project dependencies before running the live experiment."
        ) from error

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    for result in compare_prompts(client, model):
        print(f"[{result.variation}] {result.output}")


if __name__ == "__main__":
    main()