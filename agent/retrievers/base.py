"""Common interface every retrieval provider must implement.

Keeping this abstract means adding a third provider (SerpAPI, Bing, arXiv,
etc.) later is a matter of writing one new class, not touching the
orchestrator or any other node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent.types import SearchResult


class Retriever(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a search and return normalized results.

        Implementations must raise on failure (timeout, rate limit, bad
        response) rather than silently returning an empty list — the
        orchestrator is responsible for catching and logging failures so
        they can be surfaced in the final answer.
        """
        raise NotImplementedError
