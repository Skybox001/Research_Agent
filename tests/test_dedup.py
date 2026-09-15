from agent.dedup import canonicalize_url, dedup_exact, dedup_near
from agent.types import SearchResult


def _res(url, title, snippet="s"):
    return SearchResult(url=url, title=title, snippet=snippet, source="tavily", sub_query="q")


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


def test_dedup_near_removes_identical():
    text = "the quick brown fox jumps over the lazy dog"
    results = [
        _res("https://a.com/x", text, text),
        _res("https://b.com/y", text, text),
    ]
    assert len(dedup_near(results)) == 1


def test_dedup_near_keeps_different():
    results = [
        _res("https://a.com/x", "Feline behaviour", "cats dogs pets"),
        _res("https://b.com/y", "Quantum computing", "qubit entanglement error correction"),
    ]
    assert len(dedup_near(results)) == 2


def test_dedup_near_single_and_empty():
    assert dedup_near([]) == []
    single = _res("https://a.com/x", "T", "S")
    assert dedup_near([single]) == [single]
