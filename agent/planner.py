"""Decomposes a research question into sub-queries.

For simple factual questions, decomposition just returns the original
question so we don't waste an LLM call or fragment the search unnecessarily.
"""

from __future__ import annotations

import json
import logging

from agent.llm import call_llm

logger = logging.getLogger(__name__)

PLAN_PROMPT = """You are a research planning assistant.

Given a research question, decide whether it needs to be broken into
sub-questions to be answered well, or whether it can be searched as-is.

Rules:
- If the question is simple/factual/single-hop, return exactly one sub-query
  (the original question, possibly cleaned up).
- If it is multi-part, comparative, or requires background context, break it
  into 2-4 focused sub-queries.
- Sub-queries must be search-engine-friendly (short, keyword-rich).

Return ONLY a JSON array of strings. No prose, no markdown fences.

Question: {question}
"""


def _strip_code_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences despite instructions."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def plan(question: str) -> list[str]:
    prompt = PLAN_PROMPT.format(question=question)
    try:
        raw = call_llm(prompt)
    except Exception as exc:
        logger.warning("Planning LLM call failed (%s); searching the raw question", exc)
        return [question]

    raw = _strip_code_fences(raw)
    try:
        sub_queries = json.loads(raw)
        if isinstance(sub_queries, list) and all(isinstance(s, str) for s in sub_queries):
            return sub_queries[:4] or [question]
    except (json.JSONDecodeError, TypeError):
        pass
    # Fall back to the raw question if the LLM output isn't parseable JSON —
    # planning is an optimization, not a hard dependency.
    return [question]
