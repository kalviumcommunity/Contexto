from dotenv import load_dotenv
import os

load_dotenv()

print("===================================")
print("Contexto - Media Research Assistant")
print("===================================")

print("OPENAI_BASE_URL:", bool(os.getenv("OPENAI_BASE_URL")))
print("OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
print("CHAT_MODEL:", bool(os.getenv("CHAT_MODEL")))
print("EMBED_MODEL:", bool(os.getenv("EMBED_MODEL")))

print("\nWorkspace setup successful!")