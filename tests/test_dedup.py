from agent.dedup import canonicalize_url, dedup_exact
from agent.types import SearchResult


def test_canonicalize_strips_tracking_params():
    a = canonicalize_url("https://Example.com/Article/?utm_source=x&id=5")
    b = canonicalize_url("https://example.com/article?id=5")
    assert a == b


def test_canonicalize_strips_trailing_slash():
    a = canonicalize_url("https://example.com/page/")
    b = canonicalize_url("https://example.com/page")
    assert a == b


def test_dedup_exact_removes_duplicate_urls():
    results = [
        SearchResult(url="https://a.com/x", title="A", snippet="s", source="tavily", sub_query="q"),
        SearchResult(url="https://a.com/x/", title="A dup", snippet="s2", source="duckduckgo", sub_query="q"),
        SearchResult(url="https://b.com/y", title="B", snippet="s3", source="tavily", sub_query="q"),
    ]
    deduped = dedup_exact(results)
    assert len(deduped) == 2
