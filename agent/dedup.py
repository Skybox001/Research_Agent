"""URL canonicalization and near-duplicate detection.

Two layers:
1. Exact dedup on canonicalized URL (strip tracking params, trailing slash,
   scheme differences).
2. Near-duplicate dedup on title+snippet embedding similarity, since the
   same story often appears at two different URLs (syndication, mirrors).
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from agent.types import SearchResult

logger = logging.getLogger(__name__)

_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid"}

_embedder = None


def _get_embedder():
    # Imported lazily so URL canonicalization / exact dedup (no ML needed)
    # don't pay the torch/sentence-transformers import cost.
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip().lower())
    query = [(k, v) for k, v in parse_qsl(parsed.query) if k not in _TRACKING_PARAMS]
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", urlencode(query), ""))


def dedup_exact(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    deduped = []
    for r in results:
        canon = canonicalize_url(r.url)
        if canon in seen:
            continue
        seen.add(canon)
        deduped.append(r)
    return deduped


def dedup_near(results: list[SearchResult], threshold: float = 0.92) -> list[SearchResult]:
    """Drops results whose title+snippet embedding is near-identical to one
    already kept. O(n^2) but n is small (tens of results per run).

    Best-effort: if the embedding model isn't available (offline, first-run
    download failure), the exact-deduped set is returned unchanged. Dedup is
    an optimization, not a correctness gate.
    """
    if len(results) <= 1:
        return results

    try:
        import numpy as np  # lazy: only needed on the ML path

        model = _get_embedder()
        texts = [f"{r.title} {r.snippet}" for r in results]
        embeddings = model.encode(texts, normalize_embeddings=True)

        kept: list[int] = []
        for i, emb in enumerate(embeddings):
            is_dup = False
            for j in kept:
                sim = float(np.dot(emb, embeddings[j]))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(i)
        return [results[i] for i in kept]
    except Exception as exc:
        logger.warning(
            "Near-duplicate detection unavailable (%s); returning exact-deduped set", exc
        )
        return results


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    return dedup_near(dedup_exact(results))
