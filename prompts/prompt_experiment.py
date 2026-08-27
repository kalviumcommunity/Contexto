import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.answer import ANSWER, render


# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# Read configuration
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


print("===================================")
print("Contexto - Prompt Experiment")
print("===================================")

print("ENV FILE:", ENV_FILE)
print("API KEY FOUND:", bool(API_KEY))
print("BASE URL:", BASE_URL)
print("CHAT MODEL:", CHAT_MODEL)


# Check API key
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY was not found in .env"
    )


# Create client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# Test prompt
messages = [{
    "role": "user",
    "content": render(
        ANSWER,
        context="No retrieved context was provided.",
        question="Say hello in one sentence.",
    ),
}]


print("\nSending request...\n")


try:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )

    reply = response.choices[0].message.content

    print("Assistant:", reply)

    if response.usage:
        print("\nUsage:")
        print("Input tokens:", response.usage.prompt_tokens)
        print("Output tokens:", response.usage.completion_tokens)
        print("Total tokens:", response.usage.total_tokens)


except AuthenticationError as e:
    print("\nAuthentication failed.")
    print("Check that your OPENAI_API_KEY is valid.")
    print("Also check that the key belongs to the API project you are using.")
    print("\nError:", e)


except RateLimitError as e:
    print("\nRate limit or quota error.")
    print("Check your API project billing/usage limits.")
    print("\nError:", e)


except Exception as e:
    print("\nUnexpected error:")
    print(type(e).__name__, e)