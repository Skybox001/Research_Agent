from agent.types import SearchResult
from agent.verifier import check_citations, verify_evidence


def _res(url, source, title="t", snippet="s", content=None):
    return SearchResult(url=url, title=title, snippet=snippet, source=source, sub_query="q", content=content)


def test_verify_evidence_flags_single_provider():
    notes = verify_evidence(
        [_res("https://a.com/x", "tavily"), _res("https://b.com/y", "tavily")]
    )
    assert any("single provider" in n for n in notes)


def test_verify_evidence_no_single_provider_when_two_sources():
    notes = verify_evidence(
        [_res("https://a.com/x", "tavily"), _res("https://b.com/y", "duckduckgo")]
    )
    assert not any("single provider" in n for n in notes)


def test_verify_evidence_flags_single_domain():
    notes = verify_evidence(
        [_res("https://a.com/x", "tavily"), _res("https://a.com/y", "duckduckgo")]
    )
    assert any("single domain" in n for n in notes)


def test_verify_evidence_flags_snippet_only_sources():
    notes = verify_evidence(
        [
            _res("https://a.com/x", "tavily", content="full body text"),
            _res("https://b.com/y", "tavily"),
        ]
    )
    assert any("snippet-only" in n for n in notes)


def test_verify_evidence_empty():
    notes = verify_evidence([])
    assert notes  # must clearly flag an ungrounded answer


def test_check_citations_flags_out_of_range():
    notes = check_citations("Answer with [7] and [1].", 3)
    assert any("out-of-range" in n for n in notes)


def test_check_citations_flags_missing_citations():
    notes = check_citations("No bracket markers at all.", 3)
    assert any("no inline" in n for n in notes)


def test_check_citations_ok_for_valid_references():
    assert check_citations("Claim [1] and claim [2], plus [3].", 3) == []


def test_check_citations_zero_sources_no_error():
    assert check_citations("Whatever claim.", 0) == []