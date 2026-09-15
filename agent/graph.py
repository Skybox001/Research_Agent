"""Wires the pipeline together as an explicit LangGraph state graph.

Nodes are kept as thin wrappers around the pure functions in planner.py,
fetcher.py, dedup.py, ranker.py, synthesizer.py — this file only owns
sequencing, provider fan-out, and failure bookkeeping.
"""

from __future__ import annotations

import concurrent.futures
import logging

from langgraph.graph import StateGraph, END

from agent.dedup import deduplicate
from agent.fetcher import fetch_content
from agent.planner import plan
from agent.ranker import rank
from agent.retrievers.base import Retriever
from agent.synthesizer import synthesize
from agent.types import ProviderFailure, ResearchState, SearchResult
from agent.verifier import check_citations, verify_evidence

logger = logging.getLogger(__name__)


def build_graph(retrievers: list[Retriever], top_k: int = 8, fetch_top_n: int = 5):
    def plan_node(state: ResearchState) -> ResearchState:
        sub_queries = plan(state["question"])
        return {**state, "sub_queries": sub_queries}

    def search_node(state: ResearchState) -> ResearchState:
        raw_results: list[SearchResult] = []
        failures: list[ProviderFailure] = []

        def search_one(retriever: Retriever, query: str) -> tuple[list[SearchResult], list[ProviderFailure]]:
            try:
                return retriever.search(query, 5), []
            except Exception as exc:  # provider-specific errors vary widely
                logger.warning("Provider %s failed on '%s': %s", retriever.name, query, exc)
                return [], [ProviderFailure(source=retriever.name, reason=str(exc))]

        def search_sequential(retriever: Retriever, queries: list[str]) -> tuple[list[SearchResult], list[ProviderFailure]]:
            # For scrapers that aren't thread-safe (parallel_ok=False), run
            # their sub-queries in order inside a single worker instead of
            # fanning out into concurrent requests.
            results: list[SearchResult] = []
            fails: list[ProviderFailure] = []
            for query in queries:
                res, f = search_one(retriever, query)
                results.extend(res)
                fails.extend(f)
            return results, fails

        # Fan out: official APIs run each (retriever, sub-query) concurrently
        # with failure isolation per pair; unofficial scrapers are
        # sequentialized. One provider's failure never affects the others.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for retriever in retrievers:
                if getattr(retriever, "parallel_ok", True):
                    for sub_query in state["sub_queries"]:
                        fut = pool.submit(search_one, retriever, sub_query)
                        futures[fut] = retriever.name
                else:
                    fut = pool.submit(search_sequential, retriever, state["sub_queries"])
                    futures[fut] = retriever.name

            for fut in concurrent.futures.as_completed(futures):
                source = futures[fut]
                try:
                    res, f = fut.result()
                    raw_results.extend(res)
                    failures.extend(f)
                except Exception as exc:  # should not happen (all caught above), keep it from failing the run
                    logger.warning("Provider %s failed: %s", source, exc)
                    failures.append(ProviderFailure(source=source, reason=str(exc)))

        return {**state, "raw_results": raw_results, "provider_failures": failures}

    def dedup_node(state: ResearchState) -> ResearchState:
        deduped = deduplicate(state["raw_results"])
        return {**state, "deduped_results": deduped}

    def rank_node(state: ResearchState) -> ResearchState:
        ranked = rank(state["question"], state["deduped_results"], top_k=top_k)
        return {**state, "ranked_results": ranked}

    def fetch_node(state: ResearchState) -> ResearchState:
        top = state["ranked_results"][:fetch_top_n]
        rest = state["ranked_results"][fetch_top_n:]

        def safe_fetch(r: SearchResult) -> str | None:
            # fetch_content already catches HTTP/transport errors; this is a
            # final belt-and-braces so an exotic failure in one thread can
            # never take down the whole node.
            try:
                return fetch_content(r.url)
            except Exception as exc:
                logger.warning("Unexpected fetch error for %s: %s", r.url, exc)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=fetch_top_n) as pool:
            contents = list(pool.map(safe_fetch, top))
        for r, content in zip(top, contents):
            r.content = content  # None on failure — synthesizer falls back to snippet

        return {**state, "ranked_results": top + rest}

    def verify_node(state: ResearchState) -> ResearchState:
        notes = verify_evidence(state["ranked_results"])
        return {**state, "verification_notes": notes}

    def synthesize_node(state: ResearchState) -> ResearchState:
        results = state["ranked_results"]
        answer = synthesize(state["question"], results)
        references = [
            {"index": i + 1, "url": r.url, "title": r.title, "source": r.source}
            for i, r in enumerate(results)
        ]
        uncertainties: list[str] = list(state.get("verification_notes", []))
        if state.get("provider_failures"):
            uncertainties.append(
                "Providers that failed during search: "
                + ", ".join(f"{f.source} ({f.reason})" for f in state["provider_failures"])
            )
        uncertainties.extend(check_citations(answer, len(results)))
        return {**state, "answer": answer, "references": references, "uncertainties": uncertainties}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("rank", rank_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("verify", verify_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "dedup")
    graph.add_edge("dedup", "rank")
    graph.add_edge("rank", "fetch")
    graph.add_edge("fetch", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
