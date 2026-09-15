"""Shared data structures used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict


@dataclass
class SearchResult:
    """A single normalized result from any retrieval provider."""

    url: str
    title: str
    snippet: str
    source: str  # provider name, e.g. "tavily" or "duckduckgo"
    sub_query: str
    content: Optional[str] = None  # populated after fetch
    score: float = 0.0  # populated after ranking


@dataclass
class ProviderFailure:
    source: str
    reason: str


class ResearchState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""

    question: str
    sub_queries: list[str]
    raw_results: list[SearchResult]
    deduped_results: list[SearchResult]
    ranked_results: list[SearchResult]
    provider_failures: list[ProviderFailure]
    verification_notes: list[str]
    answer: str
    uncertainties: list[str]
    references: list[dict]
