from dotenv import load_dotenv
import os
import logging

from openai import OpenAI, AuthenticationError, RateLimitError

load_dotenv()

print("===================================")
print("Contexto - Media Research Assistant")
print("===================================")

print("OPENAI_BASE_URL:", bool(os.getenv("OPENAI_BASE_URL")))
print("OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
print("CHAT_MODEL:", bool(os.getenv("CHAT_MODEL")))
print("EMBED_MODEL:", bool(os.getenv("EMBED_MODEL")))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create OpenAI-compatible client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Chat messages
messages = [
    {
        "role": "system",
        "content": "You are a concise assistant."
    },
    {
        "role": "user",
        "content": "Say hello in one sentence."
    }
]

logging.info("REQUEST: %s", messages)

try:
    # Send request to chat model
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=messages
    )

    # Extract generated response
    reply = response.choices[0].message.content

    logging.info("RESPONSE: %s", reply)
    logging.info("USAGE: %s", response.usage)

    print("\nAssistant:", reply)

except AuthenticationError:
    print("\nAuth failed (401): check OPENAI_API_KEY in your .env")

except RateLimitError:
    print("\nRate limited (429): slow down and retry with backoff")

except Exception as e:
    print("\nAn error occurred:", e)