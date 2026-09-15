from __future__ import annotations

import os

from tavily import TavilyClient

from agent.retrievers.base import Retriever
from agent.types import SearchResult


class TavilyRetriever(Retriever):
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self._client = TavilyClient(api_key=key)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = self._client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
        results = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    source=self.name,
                    sub_query=query,
                )
            )
        return results
