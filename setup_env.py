"""
setup_env.py
------------
Run this ONCE to create your .env file with your Groq API key.
Usage:  python setup_env.py
Get a FREE Groq key at: https://console.groq.com
"""

import io
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows so this script never crashes on its own output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ENV_PATH = Path(__file__).parent / ".env"


def main():
    print("=" * 55)
    print("  Smart Fault Diagnosis System - Environment Setup")
    print("=" * 55)
    print()
    print("Get your FREE Groq API key at:")
    print("  https://console.groq.com  ->  API Keys  ->  Create Key")
    print()

    if ENV_PATH.exists():
        existing = ENV_PATH.read_text(encoding="utf-8").strip()
        if "GROQ_API_KEY" in existing and "your_groq" not in existing:
            print("A Groq API key is already configured in .env")
            overwrite = input("Overwrite it? (y/N): ").strip().lower()
            if overwrite != "y":
                print("Keeping existing .env. Done.")
                return

    key = input("Paste your Groq API key (starts with gsk_): ").strip()

    if not key:
        print("No key entered. Aborting.")
        return

    if not key.startswith("gsk_"):
        print("Warning: Groq keys usually start with 'gsk_'. Saving anyway.")

    ENV_PATH.write_text(f"GROQ_API_KEY={key}\n", encoding="utf-8")
    print()
    print(f"Saved to: {ENV_PATH}")
    print()
    print("Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
