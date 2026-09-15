"""Single choke point for LLM calls, so provider/model is swappable in one
place and every call goes through the same retry policy.

Uses Groq (free tier, OpenAI-compatible chat API) rather than a paid
provider, since the same choke-point pattern makes swapping providers a
one-file change if a different model is needed later.
"""

from __future__ import annotations

import os

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

MODEL = os.environ.get("RESEARCH_AGENT_MODEL", "llama-3.3-70b-versatile")

_client: Groq | None = None


def _get_client() -> Groq:
    # Built lazily so importing this module doesn't require GROQ_API_KEY to
    # already be in the environment (e.g. before main.py calls load_dotenv()).
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=key)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(prompt: str, max_tokens: int = 1024, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()
