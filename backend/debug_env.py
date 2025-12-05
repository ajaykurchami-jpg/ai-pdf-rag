import os
from dotenv import load_dotenv

# 1. Print where we are right now
print(f"Current Working Directory: {os.getcwd()}")

# 2. Try to load .env
loaded = load_dotenv()
print(f"Did .env load? {loaded}")

# 3. Check for the key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"Success! Key found: {api_key[:5]}...")
else:
    print("Failure: Key is None.")

# 4. List all files in this folder to check for typos
print("\nFiles in this folder:")
for f in os.listdir():
    print(f" - {f}")