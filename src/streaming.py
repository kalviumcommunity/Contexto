import asyncio
import json
from collections.abc import AsyncIterator


async def rag_pipeline_stream(question: str) -> AsyncIterator[dict]:
    citations = [
        {
            "id": "source-1",
            "label": "[1]",
            "document": "account-guide.md",
            "chunk_id": "chunk-1",
            "text": "Learners can reset a forgotten password from the account settings page by choosing Reset password and following the reset link."
        }
    ]

    yield {"type": "citations", "sources": citations}

    answer = (
        "A learner can reset a forgotten password from the account settings "
        "page by choosing Reset password and following the reset link. [1]"
    )

    for word in answer.split(" "):
        yield {"type": "token", "text": word + " "}
        await asyncio.sleep(0.03)

    yield {"type": "done"}


def format_sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"
