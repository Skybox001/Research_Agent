"""Ranks deduplicated results by relevance to the original question, with a
small bonus for source diversity so the shortlist isn't dominated by one
provider.

Primary path: embedding similarity to the question (all-MiniLM-L6-v2). If
the model is unavailable (offline, first-run download failure), a lexical
token-overlap fallback keeps the pipeline alive with a degraded-but-useful
ranking.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from agent.dedup import _get_embedder
from agent.types import SearchResult

logger = logging.getLogger(__name__)

DIVERSITY_BONUS = 0.05

_TERM_RE = re.compile(r"[a-z0-9']+")


def _lexical_score(question: str, r: SearchResult) -> float:
    """Token-overlap between the question and result title+snippet.
    Crude but deterministic and dependency-free."""
    question_terms = set(_TERM_RE.findall(question.lower()))
    if not question_terms:
        return 0.0
    text = f"{r.title} {r.snippet}".lower()
    return float(sum(1 for t in question_terms if t in text))


def _apply_diversity_and_topk(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    # Reward the top pick from each source so the shortlist isn't one
    # provider's opinion poll; interleaving alone doesn't guarantee this when
    # one provider simply has more/better results.
    by_source: dict[str, list[SearchResult]] = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r)
    for bucket in by_source.values():
        bucket.sort(key=lambda r: r.score, reverse=True)
        if bucket:
            bucket[0].score += DIVERSITY_BONUS

    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    return ranked[:top_k]


def rank(question: str, results: list[SearchResult], top_k: int = 8) -> list[SearchResult]:
    if not results:
        return []

    try:
        model = _get_embedder()
        query_emb = model.encode([question], normalize_embeddings=True)[0]
        doc_embs = model.encode(
            [f"{r.title} {r.snippet}" for r in results], normalize_embeddings=True
        )
        for r, emb in zip(results, doc_embs):
            r.score = float(np.dot(query_emb, emb))
    except Exception as exc:
        logger.warning("Embedding ranking unavailable (%s); using lexical fallback", exc)
        for r in results:
            r.score = _lexical_score(question, r)

    return _apply_diversity_and_topk(results, top_k)