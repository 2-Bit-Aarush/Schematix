"""
llm.py (FIXED & OPTIMIZED)
--------------------------
Improved reliability + better prompt + safer parsing.
"""

import os
import textwrap
from typing import Optional
import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {_API_KEY}",
    "Content-Type": "application/json",
}

# Better model for stability
MODEL_NAME = "meta-llama/llama-3-8b-instruct"


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def _call(prompt: str) -> Optional[str]:
    if not _API_KEY:
        print("[LLM] WARNING: OPENROUTER_API_KEY not set.")
        return None

    try:
        response = requests.post(
            BASE_URL,
            headers=HEADERS,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 200,
            },
            timeout=20,
        )

        data = response.json()

        # Debug once if needed
        # print("[LLM RAW]:", data)

        if "choices" not in data:
            print("[LLM] Invalid response:", data)
            return None

        content = data["choices"][0]["message"].get("content", "")

        if not content or not content.strip():
            print("[LLM] Empty response")
            return None

        return content.strip()

    except Exception as exc:
        print(f"[LLM] API call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. Query Enhancement
# ---------------------------------------------------------------------------

def enhance_query(query: str) -> str:
    prompt = textwrap.dedent(f"""
        Rewrite the query into a better semantic search query.

        Keep it under 25 words.
        Do NOT explain.

        Query: {query}
    """).strip()

    result = _call(prompt)

    if not result or len(result) > 200:
        return query

    print(f"[LLM] enhance_query: '{query}' → '{result}'")
    return result


# ---------------------------------------------------------------------------
# 2. Answer Generation (FIXED)
# ---------------------------------------------------------------------------

def generate_answer(query: str, context: str) -> str:
    if not context.strip():
        return "No relevant documents found."

    # HARD LIMIT context (VERY IMPORTANT)
    context = context[:800]

    prompt = textwrap.dedent(f"""
        Use the context to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer in 1-2 short sentences.
    """).strip()

    result = _call(prompt)

    if not result:
        return "Answer generation failed. Check sources."

    return result


# ---------------------------------------------------------------------------
# 3. Title Generation
# ---------------------------------------------------------------------------

def generate_title(text: str) -> str:
    preview = " ".join(text.split()[:200])

    prompt = textwrap.dedent(f"""
        Generate a short title.

        Max 8 words.
        No punctuation at end.

        Text:
        {preview}
    """).strip()

    result = _call(prompt)

    if not result or len(result) > 80:
        return ""

    return result.strip().strip("\"'")