"""Fetches full page content for a shortlist of results.

We only fetch top-ranked results, not every raw result, since full-page
fetch is the slowest and most failure-prone step (paywalls, JS-rendered
pages, timeouts). Snippet-only results still contribute to the answer if
fetch fails.
"""

from __future__ import annotations

import logging

import httpx
import trafilatura

logger = logging.getLogger(__name__)

MAX_CHARS = 4000  # cap per-document content to keep prompts bounded


def fetch_content(url: str, timeout: float = 8.0) -> str | None:
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
        )
        response.raise_for_status()
    except (httpx.HTTPError, httpx.TransportError) as exc:
        logger.warning("Fetch failed for %s: %s", url, exc)
        return None

    try:
        extracted = trafilatura.extract(response.text)
    except Exception as exc:  # extraction is best-effort; never crash the run
        logger.warning("Extraction failed for %s: %s", url, exc)
        return None
    if not extracted:
        return None
    return extracted[:MAX_CHARS]
