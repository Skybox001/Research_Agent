"""Evidence verification: checks the shortlist and the final answer for
structural weaknesses that would undermine grounding.

Two checks, both purely symbolic (no LLM involved, so they run fast and are
deterministic):

1. verify_evidence()  — ran after fetch, on the ranked shortlist. Flags
   single-provider dependence, single-domain dependence (a sign the same
   story may be mirrored/syndicated), and how much of the evidence is
   snippet-only because full content could not be fetched.

2. check_citations()  — ran after synthesis. Parses the answer's [n] markers
   and flags out-of-range indices and answers that cite nothing at all.

These are deliberately heuristic guard-rails, not a claim-truthfulness
check: the LLM still decides what is/isn't supported by the evidence. The
point is to surface structural gaps the LLM might not mention, and to catch
citation mistakes programmatically.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from agent.types import SearchResult

_CITATION_RE = re.compile(r"\[(\d+)\]")

_DOMAIN_OR_PROVIDER_FLAG = "single provider"
_SNIPPET_ONLY_FLAG = "snippet-only"


def _providers(results: list[SearchResult]) -> set[str]:
    return {r.source for r in results if r.source}


def _domains(results: list[SearchResult]) -> set[str]:
    return {urlparse(r.url).netloc for r in results if r.url}


def verify_evidence(results: list[SearchResult]) -> list[str]:
    """Return a list of grounding-risk notes about the evidence shortlist."""
    notes: list[str] = []

    if not results:
        return ["No evidence survived retrieval; the answer is ungrounded."]

    providers = _providers(results)
    if len(providers) < 2:
        names = ", ".join(sorted(providers)) or "none"
        notes.append(
            f"Evidence comes from a {_DOMAIN_OR_PROVIDER_FLAG} ({names}) — "
            "cross-checking against an independent source is limited."
        )

    domains = _domains(results)
    if len(domains) < 2:
        names = ", ".join(sorted(domains)) or "none"
        notes.append(
            "All evidence URLs share a single domain (%s); if the story is "
            "syndicated this set may be less diverse than it looks."
            % names
        )

    fetched = [r for r in results if r.content]
    if len(fetched) < len(results):
        gap = len(results) - len(fetched)
        notes.append(
            f"{gap} of {len(results)} sources are {_SNIPPET_ONLY_FLAG} "
            "(full content could not be fetched); those claims rest on the "
            "search-result snippet alone."
        )

    return notes


def check_citations(answer: str, n_sources: int) -> list[str]:
    """Return notes about citation issues in a synthesized answer."""
    if n_sources <= 0:
        return []

    cited = {int(m) for m in _CITATION_RE.findall(answer)}
    valid = set(range(1, n_sources + 1))
    notes: list[str] = []

    out_of_range = cited - valid
    if out_of_range:
        notes.append(
            f"Answer cites out-of-range reference indices "
            f"{sorted(out_of_range)} (evidence had {n_sources} sources)."
        )

    if not cited:
        notes.append(
            "Answer contains no inline [n] citations — claims are not "
            "explicitly grounded in the provided evidence."
        )

    return notes