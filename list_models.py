import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("--- Attempting to list available Google AI Models ---")

try:
    # The library will automatically use the credentials set by 'gcloud auth'
    # but we can configure the key as a fallback.
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    print("\nModels that support 'generateContent':")
    for m in genai.list_models():
        # Check if the 'generateContent' method is supported
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")

except Exception as e:
    print(f"\nAn error occurred while trying to list models: {e}")

print("\n--- Script finished ---")