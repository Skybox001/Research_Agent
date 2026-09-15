from unittest.mock import patch

from agent.ranker import rank
from agent.types import SearchResult


def _res(url, source, title="t", snippet="s"):
    return SearchResult(url=url, title=title, snippet=snippet, source=source, sub_query="q")


def test_rank_lexical_default_does_not_need_embedder():
    results = [
        _res("https://a.com/llm", "tavily", title="Long context windows for LLMs", snippet="context"),
        _res("https://b.com/recipe", "duckduckgo", title="Cooking pasta recipes", snippet="pasta"),
    ]
    with patch(
        "agent.ranker._rank_embedding",
        side_effect=AssertionError("embedding must not run by default"),
    ):
        ranked = rank("long context window LLM", results, top_k=2)
    assert ranked[0].url == "https://a.com/llm"
    assert len(ranked) <= 2


def test_rank_falls_back_lexically_when_embedding_fails(monkeypatch):
    monkeypatch.setenv("RESEARCH_AGENT_EMBEDDINGS", "1")
    results = [
        _res("https://a.com/llm", "tavily", title="Long context windows for LLMs", snippet="context"),
        _res("https://b.com/recipe", "duckduckgo", title="Cooking pasta recipes", snippet="pasta"),
    ]
    with patch("agent.ranker._rank_embedding", side_effect=RuntimeError("offline")):
        ranked = rank("long context window LLM", results, top_k=2)
    assert ranked[0].url == "https://a.com/llm"
    assert len(ranked) <= 2


def test_rank_empty_input_returns_empty():
    assert rank("any question", []) == []


def test_rank_keeps_top_k():
    results = [
        _res(f"https://site{i}.com/x", "duckduckgo", title=f"story {i}", snippet="news")
        for i in range(20)
    ]
    ranked = rank("story", results, top_k=5)
    assert len(ranked) == 5