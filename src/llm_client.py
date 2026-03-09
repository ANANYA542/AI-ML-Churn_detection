"""Thin wrapper over the Groq API for retention-strategy prompts."""

import os
from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:
    st = None

from groq import Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    if st is not None:
        try:
            return st.secrets["GROQ_API_KEY"]
        except Exception:
            return None
    return None


def get_llm_response(prompt: str, system_prompt: str = "") -> str:
    """Send prompt to Groq and return the assistant text.

    Raises RuntimeError if the API key is missing.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to .env or .streamlit/secrets.toml."
        )

    client = Groq(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    sample = get_llm_response(
        "A telecom customer on month-to-month contract with high monthly charges "
        "and 3-month tenure is flagged high-risk. Suggest 3 retention actions.",
        system_prompt="You are a customer retention strategist. Be concise and practical.",
    )
    print(sample)
