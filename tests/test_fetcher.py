from unittest.mock import patch

import httpx

from agent.fetcher import fetch_content


def test_fetch_returns_none_on_connect_error():
    with patch("agent.fetcher.httpx.get", side_effect=httpx.ConnectError("boom")):
        assert fetch_content("https://example.com") is None


def test_fetch_returns_none_on_timeout():
    with patch("agent.fetcher.httpx.get", side_effect=httpx.TimeoutException("slow")):
        assert fetch_content("https://example.com") is None


def test_fetch_returns_none_on_http_error():
    resp = httpx.Response(500, request=httpx.Request("GET", "https://example.com"))
    with patch("agent.fetcher.httpx.get", return_value=resp):
        assert fetch_content("https://example.com") is None


def test_fetch_keeps_ok_content():
    html = "<html><body><article><h1>Title</h1><p>Some body text.</p></article></body></html>"
    resp = httpx.Response(200, text=html, request=httpx.Request("GET", "https://example.com"))
    with patch("agent.fetcher.httpx.get", return_value=resp):
        content = fetch_content("https://example.com")
    assert content is not None
    assert "Some body text" in content