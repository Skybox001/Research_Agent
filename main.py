"""CLI entry point.

Usage:
    python main.py "What are the tradeoffs between RAG and long-context LLMs?"
"""

from __future__ import annotations

import sys
import logging

from dotenv import load_dotenv

from agent.graph import build_graph
from agent.retrievers.duckduckgo_retriever import DuckDuckGoRetriever
from agent.retrievers.tavily_retriever import TavilyRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python main.py "<research question>"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    # Each retriever is instantiated independently; if one fails to even
    # initialize (missing key), the other still runs — degrade, don't crash.
    retrievers = []
    try:
        retrievers.append(TavilyRetriever())
    except RuntimeError as exc:
        print(f"[warn] Tavily unavailable: {exc}")
    retrievers.append(DuckDuckGoRetriever())  # no key required

    if not retrievers:
        print("[error] No retrievers available. Set TAVILY_API_KEY or check network access.")
        sys.exit(1)

    app = build_graph(retrievers)
    result = app.invoke({"question": question})

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== REFERENCES ===\n")
    for ref in result["references"]:
        print(f"[{ref['index']}] ({ref['source']}) {ref['title']} — {ref['url']}")

    if result.get("uncertainties"):
        print("\n=== NOTES / UNCERTAINTIES ===\n")
        for note in result["uncertainties"]:
            print(f"- {note}")


if __name__ == "__main__":
    main()
