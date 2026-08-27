"""Keep a multi-turn chat within a measured token budget."""

from dataclasses import dataclass
from types import SimpleNamespace

import tiktoken

from token_cost_estimator import count_tokens


SYSTEM_MESSAGE = (
    "You are Contexto, a concise internal research assistant. "
    "Use only the conversation and retrieved evidence."
)


@dataclass(frozen=True)
class RequestLog:
    turn: int
    naive_tokens: int
    request_tokens: int
    removed_turns: int


class Conversation:
    """Store messages and trim the oldest complete turns before each request."""

    def __init__(self, encoder: tiktoken.Encoding, token_budget: int) -> None:
        self.encoder = encoder
        self.token_budget = token_budget
        self.history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
        self.naive_history_tokens = self.total_tokens()
        self.request_logs: list[RequestLog] = []

    def total_tokens(self) -> int:
        """Count every message content currently held in history."""
        return sum(count_tokens(self.encoder, message["content"]) for message in self.history)

    def _trim_old_turns(self) -> int:
        removed_turns = 0
        while self.total_tokens() > self.token_budget and len(self.history) > 2:
            del self.history[1:3]
            removed_turns += 1
        if self.total_tokens() > self.token_budget:
            raise ValueError("The system message and current turn exceed the token budget.")
        return removed_turns

    def ask(self, client: "DemoClient", user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        self.naive_history_tokens += count_tokens(self.encoder, user_message)
        naive_tokens = self.naive_history_tokens
        removed_turns = self._trim_old_turns()
        request_tokens = self.total_tokens()
        response = client.chat.completions.create(messages=list(self.history))
        answer = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        self.naive_history_tokens += count_tokens(self.encoder, answer)
        self._trim_old_turns()
        self.request_logs.append(
            RequestLog(
                turn=len(self.request_logs) + 1,
                naive_tokens=naive_tokens,
                request_tokens=request_tokens,
                removed_turns=removed_turns,
            )
        )
        return answer


class DemoClient:
    """Deterministic stand-in for a chat API used by the sample run."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)

    def create(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        turn_number = sum(message["role"] == "user" for message in messages)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"Acknowledged turn {turn_number}.")
                )
            ]
        )


def main() -> None:
    encoder = tiktoken.get_encoding("cl100k_base")
    conversation = Conversation(encoder=encoder, token_budget=100)
    client = DemoClient()
    long_question = (
        "Review the archived interview notes and explain how the newsroom's source "
        "verification process changed after the publication."
    )

    print("Tokenizer: cl100k_base")
    print(f"Token budget per request: {conversation.token_budget}")
    print("turn | naive tokens | request tokens | removed old turns")
    for _ in range(8):
        conversation.ask(client, long_question)
        log = conversation.request_logs[-1]
        print(
            f"{log.turn:>4} | {log.naive_tokens:>12} | {log.request_tokens:>14} | "
            f"{log.removed_turns:>18}"
        )
    print(f"System preserved: {conversation.history[0]['role'] == 'system'}")
    print(f"Final history tokens: {conversation.total_tokens()}")


if __name__ == "__main__":
    main()