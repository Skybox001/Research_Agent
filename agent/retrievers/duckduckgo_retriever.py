from __future__ import annotations

from ddgs import DDGS

from agent.retrievers.base import Retriever
from agent.types import SearchResult


class DuckDuckGoRetriever(Retriever):
    """Unofficial scraper over DuckDuckGo's web front-ends.

    Not safe to run concurrently from many threads (the underlying
    ddgs/primp curl session isn't thread-safe), so parallel_ok = False to
    make the orchestrator sequentialize this provider's sub-queries.
    """

    name = "duckduckgo"
    parallel_ok = False

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        url=item.get("href", ""),
                        title=item.get("title", ""),
                        snippet=item.get("body", ""),
                        source=self.name,
                        sub_query=query,
                    )
                )
        return results
