"""Synthesizes a grounded answer from ranked, fetched results.

The prompt is deliberately strict: the model may only use the provided
evidence, must cite by index, and must call out gaps rather than filling
them in from parametric knowledge. This is the main hallucination-reduction
lever in the system (the other is: only pass evidence that survived ranking
and dedup, not raw search noise).
"""

from __future__ import annotations

import logging

from agent.llm import call_llm
from agent.types import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful research analyst. You answer ONLY using
the numbered evidence sources you are given. You never use outside
knowledge to fill gaps. Every factual claim must carry a bracketed citation
like [1] or [2] referring to the evidence list. If the evidence is
insufficient, conflicting, or silent on part of the question, say so
explicitly in an "Uncertainties" section rather than guessing."""

ANSWER_PROMPT = """Research question: {question}

Evidence sources:
{evidence_block}

Write:
1. A concise answer (2-6 sentences) to the question, with inline [n] citations.
2. A "Key supporting claims" list, each with its citation.
3. An "Uncertainties" section: note any conflicting sources, gaps, or claims
   you could not verify with the given evidence. If none, say "None identified."
"""


def _build_evidence_block(results: list[SearchResult]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        body = r.content or r.snippet
        lines.append(f"[{i}] Source: {r.source} | URL: {r.url}\nTitle: {r.title}\nContent: {body}\n")
    return "\n".join(lines)


def synthesize(question: str, results: list[SearchResult]) -> str:
    if not results:
        return (
            "No usable evidence was retrieved for this question (all providers "
            "failed or returned nothing relevant). Answer cannot be grounded."
        )
    evidence_block = _build_evidence_block(results)
    prompt = ANSWER_PROMPT.format(question=question, evidence_block=evidence_block)
    try:
        return call_llm(prompt, max_tokens=1200, system=SYSTEM_PROMPT)
    except Exception as exc:
        # Last-resort degradation: if the LLM is down/rate-limited after all
        # retries, still return the evidence so the run produces something
        # verifiable instead of crashing with no output at all.
        logger.warning("Synthesis LLM call failed (%s); returning raw evidence", exc)
        references = "\n".join(
            f"[{i}] {r.title} — {r.url}" for i, r in enumerate(results, start=1)
        )
        return (
            f"Synthesis failed after retries ({exc}). The following evidence "
            f"was retrieved and is shown unprocessed for manual review:\n\n{references}"
        )
