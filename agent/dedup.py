"""URL canonicalization and near-duplicate detection.

Two layers:
1. Exact dedup on canonicalized URL (strip tracking params, trailing slash,
   scheme differences).
2. Near-duplicate dedup. The default is a dependency-free word-shingle
   (Jaccard) comparison so the pipeline runs with zero heavyweight ML
   imports. Set RESEARCH_AGENT_EMBEDDINGS=1 to use all-MiniLM-L6-v2
   embedding similarity instead — better on paraphrase-style duplicates, but
   it pulls in torch/sklearn/pandas/pyarrow, which are heavy and can be
   fragile on very new Python versions (a native crash there cannot be
   caught by Python, so it's off by default).
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from agent.types import SearchResult

logger = logging.getLogger(__name__)

_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid"}

_embedder = None

# Word-shingle size and Jaccard threshold for the dependency-free near-dup.
_SHINGLE_N = 5
_SHINGLE_THRESHOLD = 0.75


def _embedding_enabled() -> bool:
    return os.environ.get("RESEARCH_AGENT_EMBEDDINGS", "").strip().lower() in {"1", "true", "yes"}


def _get_embedder():
    """Lazy singleton for the optional embedding model (opt-in only)."""
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


def _word_shingles(text: str, n: int = _SHINGLE_N) -> set[str]:
    """Character-safe word shingles: lowercase, collapse punctuation/space."""
    norm = " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())
    words = norm.split(" ")
    if len(words) < n:
        return {norm}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def dedup_near(results: list[SearchResult], threshold: float = _SHINGLE_THRESHOLD) -> list[SearchResult]:
    """Drops results whose title+snippet is near-identical to one already
    kept, using word-shingle Jaccard. Pure Python, no ML imports. O(n^2) but
    n is small (tens of results per run)."""
    if len(results) <= 1:
        return results

    kept: list[SearchResult] = []
    kept_shingles: list[set[str]] = []
    for r in results:
        shingles = _word_shingles(f"{r.title} {r.snippet}")
        if any(_jaccard(shingles, k) >= threshold for k in kept_shingles):
            continue
        kept.append(r)
        kept_shingles.append(shingles)
    return kept


def dedup_near_embedding(results: list[SearchResult], threshold: float = 0.92) -> list[SearchResult]:
    """Embedding-similarity near-dup (opt-in via RESEARCH_AGENT_EMBEDDINGS=1)."""
    if len(results) <= 1:
        return results

    import numpy as np  # lazy: only needed on the embedding path

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


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    exact = dedup_exact(results)
    if _embedding_enabled():
        try:
            return dedup_near_embedding(exact)
        except Exception as exc:
            logger.warning("Embedding near-dup unavailable (%s); using shingle near-dup", exc)
    return dedup_near(exact)