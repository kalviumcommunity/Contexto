import os
import logging

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

# Load environment variables from .env
load_dotenv()


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")


# --------------------------------------------------
# Display configuration status
# --------------------------------------------------

print("===================================")
print("Contexto - Media Research Assistant")
print("===================================")

print("OPENAI_BASE_URL:", bool(BASE_URL))
print("OPENAI_API_KEY:", bool(API_KEY))
print("CHAT_MODEL:", bool(CHAT_MODEL))
print("EMBED_MODEL:", bool(EMBED_MODEL))


# --------------------------------------------------
# Validate required configuration
# --------------------------------------------------

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. "
        "Please set a valid API key in your .env file."
    )

if not CHAT_MODEL:
    raise RuntimeError(
        "CHAT_MODEL is missing. "
        "Please set CHAT_MODEL in your .env file."
    )


# --------------------------------------------------
# Configure logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# --------------------------------------------------
# Create OpenAI client
# --------------------------------------------------

client_args = {
    "api_key": API_KEY
}

# Use custom base URL only if one is provided
if BASE_URL:
    client_args["base_url"] = BASE_URL

client = OpenAI(**client_args)


# --------------------------------------------------
# Chat messages
# --------------------------------------------------

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


# --------------------------------------------------
# Send request
# --------------------------------------------------

try:

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )

    # Extract response
    reply = response.choices[0].message.content

    logging.info("RESPONSE: %s", reply)
    logging.info("USAGE: %s", response.usage)

    print("\nAssistant:", reply)


# --------------------------------------------------
# Error handling
# --------------------------------------------------

except AuthenticationError:
    print("\nAuthentication failed (401).")
    print("Please check that OPENAI_API_KEY in your .env is valid.")

except RateLimitError:
    print("\nRate limited (429).")
    print("Please check your API usage/quota and try again later.")

except Exception as e:
    logging.exception("Unexpected error occurred.")
    print("\nAn error occurred:", e)