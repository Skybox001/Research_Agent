# Multi-Source Web Research Agent

A research agent that answers a natural-language question by querying two
independent search providers, deduplicating and ranking the results,
fetching full content for the best candidates, and synthesizing an answer
that is grounded in that evidence — with citations, and explicit
uncertainty when the evidence doesn't fully support an answer.

## Problem understanding

The brief asks for more than a "search → paste into prompt" chain. The
parts that actually matter for a system like this are: using more than one
retrieval source so a single provider's blind spots or outages don't sink
the answer, treating duplicate/near-duplicate results as noise rather than
letting them dominate the evidence set, being explicit about what the
evidence does and doesn't support, and failing gracefully — a provider
timeout or empty result set should degrade the answer's confidence, not
crash the run.

## Architecture

```
question
   │
   ▼
 plan            → LLM decomposes into 1-4 sub-queries (or passes through
   │                the original question for simple factual asks)
   ▼
 search           → Tavily + DuckDuckGo queried concurrently, one thread
   │                 per (provider, sub-query) pair. Failures are caught
   │                 per-pair and recorded, not raised.
   ▼
 dedup            → (1) exact: canonicalize URLs (strip tracking params,
   │                 trailing slash, scheme) and drop exact repeats.
   │                 (2) near-duplicate: embed title+snippet with
   │                 all-MiniLM-L6-v2 and drop anything >0.92 cosine
   │                 similarity to an already-kept result.
   ▼
 rank             → embedding similarity of each result to the *original*
   │                 question (not the sub-query), plus a small bonus for
   │                 the top result from each source so one provider can't
   │                 crowd out the other. Top-k kept (default 8).
▼
 fetch            → full-page content is fetched (via trafilatura) only for
   │                 the top N ranked results (default 5) — the rest rely
   │                 on snippet text. This bounds the slowest, most
   │                 failure-prone step. Fetch failures fall back to
   │                 snippet, they don't drop the result.
   ▼
 verify           → structural grounding checks on the shortlist: is there
   │                 more than one provider? more than one domain? how much
   │                 of the evidence is snippet-only because fetching failed?
   │                 Deterministic, no LLM call. Findings feed "Uncertainties".
   ▼
 synthesize       → LLM answers using ONLY the evidence block, with a
                     system prompt that forbids outside knowledge and
                     requires inline [n] citations plus an explicit
                     "Uncertainties" section. After synthesis the answer's
                     [n] markers are parsed and out-of-range / missing
                     citations are flagged (verifier.check_citations).
 ```

Implemented as an explicit [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph` (`agent/graph.py`) rather than one long function, so each stage
is independently testable and a new stage (e.g. a verification pass) can be
inserted without touching the others.

### Why these two sources

- **Tavily** — purpose-built for LLM research agents; returns cleaned
  content snippets and its own relevance ranking, which raises the floor on
  result quality.
- **DuckDuckGo** (via the `ddgs` package) — free, no API key, and a
  genuinely independent index from Tavily's backend, so it adds real source
  diversity rather than querying the same underlying search engine twice.

Adding a third provider (SerpAPI, Bing, arXiv, etc.) means writing one class
that implements `agent/retrievers/base.py`'s `Retriever` interface — nothing
else in the pipeline needs to change.

### Deduplication

Two layers, because URL-level dedup alone misses syndicated/mirrored
content, and embedding-only dedup alone is too slow/blunt to run on raw,
un-canonicalized results:

1. **Exact** — canonicalize the URL (lowercase, strip `utm_*`/`ref`/`fbclid`
   params, drop trailing slash) and drop exact repeats. Cheap, catches the
   common case (same article returned by both providers).
2. **Near-duplicate** — embed `title + snippet` and drop anything above a
   0.92 cosine-similarity threshold to a result already kept. Catches
   syndicated copies living at different URLs. Runs after exact dedup so
   the O(n²) comparison only has to look at the smaller, already-deduped
   set.

### Conflict handling

The system does not attempt to resolve factual conflicts between sources
algorithmically — that's a judgment call best left to the LLM synthesis
step, which sees all surviving evidence side by side and is explicitly
instructed to surface disagreement in the "Uncertainties" section rather
than silently picking a side.

### Verification

A structural, model-free check (`agent/verifier.py`) runs at two points:

1. After fetch, `verify_evidence()` inspects the shortlist and flags
   grounding risks that could otherwise go unmentioned: evidence from a
   single provider only, all URLs on a single domain (the same story may be
   syndicated/mirrored), and how much of the evidence is snippet-only
   because full content failed to fetch.
2. After synthesis, `check_citations()` parses the answer's `[n]` markers
   and flags out-of-range citation indices and answers that cite nothing at
   all — catching the "model printed a citation that wasn't in the
   evidence" failure mode programmatically.

Findings are appended to the output's "Uncertainties" notes, so they
surface to the user without requiring the LLM to volunteer them.

### Failure handling

- Each `(provider, sub-query)` search runs in its own thread; an exception
  in one is caught, logged, and recorded as a `ProviderFailure` — it does
  not affect the other provider or other sub-queries.
- If a provider fails to *initialize* (e.g. missing `TAVILY_API_KEY`), the
  CLI drops it and continues with whatever's left, rather than crashing at
  startup.
- Per-request timeouts on both search and fetch (8s default) prevent a
  single slow host from stalling the whole run. Fetching also catches
  transport-layer errors (DNS failures, refused connections) and, as a
  belt-and-braces, the fetch node runs each URL inside its own try/except —
  one bad host can never take down the whole run.
- LLM calls (`agent/llm.py`) retry up to 3 times with exponential backoff
  via `tenacity`, covering transient rate limits (Groq's free tier has
  fairly tight per-minute limits, so this matters in practice). If retries
  are exhausted: the planner falls back to searching the raw question, and
  the synthesizer returns the evidence unprocessed with an explanatory
  note, so a dead LLM degrades the answer rather than aborting the run.
- The embedding model is treated as an enhancement, not a dependency:
  ranking falls back to lexical token-overlap scoring, and near-duplicate
  detection is skipped (exact-URL dedup still applies) if the model is
  unavailable or its download fails.
- If **all** providers fail or return nothing, the synthesizer returns an
  explicit "no usable evidence" message instead of hallucinating an answer.
- Any provider failures that did occur are surfaced to the user in the
  final output as a note, even if the run still produced an answer from
  the surviving provider — plus the verifier's grounding notes and any
  citation problems found in the answer.

### Hallucination reduction

The main lever is scope control on the synthesis prompt: the system prompt
forbids using outside/parametric knowledge, requires every claim to carry a
citation index, and requires an explicit uncertainty section. The secondary
lever is upstream — only ranked, deduplicated evidence reaches the prompt,
not raw search noise, which keeps the evidence block small and relevant
enough that the model doesn't need to reach outside it to fill gaps.

## Setup and execution

```bash
git clone <this-repo>
cd research-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY (required, free at console.groq.com) and
# TAVILY_API_KEY (optional — the agent still runs on DuckDuckGo alone
# if this is missing)

python main.py "What are the tradeoffs between RAG and long-context LLMs?"
```

Run tests — the pure-logic tests need no API keys and no model download:

```bash
pytest tests/
```

## Technology choices

| Choice | Why |
|---|---|
| LangGraph | Explicit state graph maps directly onto the required separation of search/fetch/rank/verify/synthesize into distinct modules, and makes the control flow inspectable/testable node by node. |
| Tavily + DuckDuckGo | See "Why these two sources" above. |
| `all-MiniLM-L6-v2` | Small, fast, CPU-friendly embedding model — reused for both near-dup detection and ranking so there's only one embedding model to load. |
| `trafilatura` | Purpose-built for boilerplate-free article extraction, more robust than naive BeautifulSoup text scraping across arbitrary sites. |
| `tenacity` | Declarative retry/backoff for LLM calls without hand-rolled loops. |
| Groq (Llama 3.3 70B) | Used for both planning and synthesis. Free tier, no card required, and fast inference. A single `agent/llm.py` choke point makes swapping to a different provider/model a one-file change. |

## API limits, costs, and assumptions

- **Groq (LLM)** — free tier with per-minute and per-day token caps; the
  default is single-digit requests per minute, so consecutive runs can hit
  `429`s. The `tenacity` retry with exponential backoff handles burst
  cases; sustained tests should be spaced out.
- **Tavily** — a dev key ships with a monthly credit allowance
  (hundreds of free searches); `search_depth="advanced"` costs more per
  call than "basic". The agent is designed to run without Tavily at all
  (DuckDuckGo alone), so hitting the cap degrades gracefully to a
  single-source run with an explicit note.
- **DuckDuckGo** — no API key, but it's an unofficial endpoint (`ddgs`
  scrapes the web UI); it can throttle or challenge automated clients, and
  it's out of our control. That's part of why it's paired with a second,
  keyed provider rather than trusted alone.
- **Embedding model download** — `all-MiniLM-L6-v2` is fetched from SF on
  first use (~90 MB). First run needs network; afterwards it's cached and
  both dedup and ranking work fully offline.
- **Assumptions** — the question is asked in English; sources are assumed to
  be publicly fetchable over HTTP; results describe a point-in-time state of
  the web (no freshness/recency weighting).
- **Credentials** — `.env` is git-ignored and `.env.example` ships with
  placeholder strings only; this project has never had a real key in a
  committed file.

## Known limitations

- Conflict *detection* across sources is left entirely to the LLM's reading
  of the evidence block; there's no structured claim-extraction/comparison
  step, so subtle numeric or date conflicts across sources may be missed.
- Near-duplicate detection uses a fixed 0.92 threshold; it hasn't been
  tuned against a labeled dataset and may occasionally over- or
  under-merge.
- `trafilatura` extraction can fail silently on heavily JS-rendered pages
  (SPAs); the pipeline falls back to the search snippet, but this can be a
  noticeably weaker piece of evidence for such pages.
- No caching layer — every run re-queries both providers and re-fetches
  content, even for a repeated question.
- No persistent storage; each invocation is stateless.

## Possible future improvements

- A structured claim-extraction pass before synthesis, so conflicting
  claims can be flagged programmatically rather than relying on the LLM to
  notice them in free text.
- Query-level caching (provider results and fetched content) with a TTL.
- A confidence score per answer, derived from evidence agreement and
  source count, rather than a binary "uncertainties" note.
- Support for a third provider class (e.g. an academic source like
  Semantic Scholar) for research questions that skew technical/academic.
- Async I/O instead of thread pools for the search/fetch fan-out.

## Testing and evaluation

The test suite (`tests/`) runs with no API keys and no model download:

- `test_dedup.py` — URL canonicalization (tracking-param stripping, trailing
  slashes, scheme/domain normalization) and exact-URL dedup.
- `test_planner.py` — JSON parsing, markdown-fence stripping, and graceful
  fallback when the LLM crashes or returns garbage.
- `test_fetcher.py` — transport/HTTP/timeout errors return `None` (never
  crash the run); happy path extracts body text.
- `test_ranker.py` — lexical fallback behaves sanely when the embedding
  model is missing, respects `top_k`, handles empty input.
- `test_verifier.py` — single-provider / single-domain / snippet-only
  grounding notes, and citation out-of-range / missing checks.

The LLM-dependent stages (real planning, synthesis) and the retrievers are
not unit-tested against live endpoints (they need keys/network and are rate
limited); they're exercised end-to-end via `python main.py "..."`, shown in
the demo video. Any provider/LLM failure in that path degrades with an
explicit note rather than crashing, which is itself part of the
evaluation.

## Implementation notes

All modules (`planner`, `retrievers/*`, `dedup`, `ranker`, `fetcher`,
`verifier`, `synthesizer`, `graph`, `llm`) and the CLI (`main.py`) were
written for this submission. Pure-logic components (URL canonicalization,
exact dedup, planner failure/fallback, fetch error handling, ranker
fallback, verifier checks) are covered by unit tests in `tests/` that run
without API keys; the embedding path, retrievers, and LLM synthesis require
network access / API keys and are exercised via the end-to-end CLI run shown
in the demo video.
