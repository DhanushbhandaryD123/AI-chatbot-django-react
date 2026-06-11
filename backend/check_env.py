import os
from dotenv import load_dotenv

# Load the .env file from the current directory
load_dotenv()

key = os.getenv("OPENAI_API_KEY")

print("\n--- Environment Check ---")
if key:
    # Only show the first 8 characters for security
    print(f"✅ SUCCESS: API Key found!")
    print(f"Key starts with: {key[:8]}...")
    print(f"Key length: {len(key)} characters")
else:
    print("❌ ERROR: OPENAI_API_KEY not found.")
    print(f"Current Directory: {os.getcwd()}")
    print("Files in this directory:", os.listdir('.'))
print("--------------------------\n")