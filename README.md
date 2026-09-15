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
 search           → Tavily + DuckDuckGo. Official APIs (Tavily) fan out one
   │                 thread per (provider, sub-query), isolated per pair;
   │                 scrapers (DuckDuckGo) declare `parallel_ok=False` and
   │                 run their sub-queries sequentially in one worker.
   ▼
 dedup            → (1) exact: canonicalize URLs (strip tracking params,
   │                 trailing slash, scheme) and drop exact repeats.
   │                 (2) near-duplicate: word-shingle Jaccard on
   │                 title+snippet by default (dependency-free); use
   │                 all-MiniLM-L6-v2 cosine (>0.92) instead with
   │                 RESEARCH_AGENT_EMBEDDINGS=1.
   ▼
 rank             → lexical token-overlap with the *original* question (not
   │                 the sub-query) plus a small bonus for the top result
   │                 from each source so one provider can't crowd out the
   │                 other. Embedding similarity replaces the lexical score
   │                 only when embeddings are enabled. Top-k kept (default 8).
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
2. **Near-duplicate** — by default, word-shingle (5-gram) Jaccard on
   `title + snippet` drops anything ≥0.75 similar to a result already kept.
   Pure Python and dependency-free. With `RESEARCH_AGENT_EMBEDDINGS=1`,
   `title + snippet` is embedded with all-MiniLM-L6-v2 and anything >0.92
   cosine similarity is dropped instead — better on paraphrase-style dupes
   at the cost of heavy native deps. Runs after exact dedup so the O(n²)
   comparison only has to look at the smaller, already-deduped set.

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

- Official API providers (Tavily) run each `(provider, sub-query)` search in
  its own thread; an exception in one is caught, logged, and recorded as a
  `ProviderFailure` — it does not affect the provider, other sub-queries, or
  the other provider.
- Scraper providers (DuckDuckGo via `ddgs`) are not thread-safe: they wrap a
  curl-based session, and hammering them with concurrent requests can
  corrupt the process heap and crash later in unrelated native code. They
  declare `parallel_ok = False` and the orchestrator runs their sub-queries
  sequentially in a single worker instead. It's also more polite to the
  unofficial endpoint.
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
- The embedding model is *off by default* (`RESEARCH_AGENT_EMBEDDINGS=1`
  enables it). The default near-duplicate detection is a dependency-free
  word-shingle Jaccard comparison, and the default ranking is lexical
  token-overlap scoring — pure Python, no `torch`/`sklearn`/`pandas`/
  `pyarrow`. This matters because those native packages can segfault (not
  just raise) on very new Python builds, and a native crash is *not*
  catchable by `try/except`; keeping them off the critical path makes the
  pipeline robust on machines where they misbehave. When the opt-in
  embedding path fails with a *Python* exception, dedup/ranking fall back
  to the dependency-free versions automatically.
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
| Dedup/rank (default) | Word-shingle Jaccard near-dup + lexical token-overlap ranking — pure Python, no native ML deps on the critical path, so the pipeline is robust even on very new Python builds. |
| `all-MiniLM-L6-v2` (opt-in) | When `RESEARCH_AGENT_EMBEDDINGS=1`, a small CPU-friendly embedding model sharpens both near-dup detection and ranking; one model reused by both stages. |
| `trafilatura` | Purpose-built for boilerplate-free article extraction, more robust than naive BeautifulSoup text scraping across arbitrary sites. |
| `tenacity` | Declarative retry/backoff for LLM calls without hand-rolled loops. |
| Groq (Llama 3.3 70B / GPT-OSS) | Used for both planning and synthesis. Free tier, no card required, and fast inference. A single `agent/llm.py` choke point makes swapping to a different provider/model a one-file change. |

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
- **Embedding model (optional)** — `all-MiniLM-L6-v2` is only downloaded
  when `RESEARCH_AGENT_EMBEDDINGS=1` is set (~90 MB from SF on first use).
  The default dependency-free path needs no model download and works fully
  offline.
- **Environment/Python caveat** — heavy native ML packages (`torch`,
  `sklearn`, `pandas`, `pyarrow`) can crash with a hard segfault on very
  new Python versions (3.14+); a native crash cannot be caught in Python.
  Because embeddings are opt-in, a machine where those packages misbehave
  still runs the full pipeline on the dependency-free default.
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
- Near-duplicate detection uses fixed thresholds (shingle Jaccard ≥0.75 by
  default, embedding cosine >0.92 when opted in); neither has been tuned
  against a labeled dataset, so both may occasionally over- or under-merge.
- The dependency-free lexical ranking is a token-overlap heuristic; it's
  weaker than semantic similarity at matching paraphrased-but-relevant
  results, which is why the embedding path exists as an opt-in.
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
  slashes, scheme/domain normalization), exact-URL dedup, and shingle-based
  near-duplicate detection.
- `test_planner.py` — JSON parsing, markdown-fence stripping, and graceful
  fallback when the LLM crashes or returns garbage.
- `test_fetcher.py` — transport/HTTP/timeout errors return `None` (never
  crash the run); happy path extracts body text.
- `test_ranker.py` — the dependency-free lexical path by default (embedding
  path is *not* invoked without the opt-in), clean lexical fallback when the
  embedding path errors, `top_k` respected, empty input handled.
- `test_verifier.py` — single-provider / single-domain / snippet-only
  grounding notes, and citation out-of-range / missing checks.

The LLM-dependent stages (real planning, synthesis) and the retrievers are
not unit-tested against live endpoints (they need keys/network and are rate
limited); they're exercised end-to-end via `python main.py "..."`. The
end-to-end run also doubles as a failure-injection test in practice:
provider/LLM/fetch failures in that path degrade with an explicit note
rather than crashing.

## Implementation notes

All modules (`planner`, `retrievers/*`, `dedup`, `ranker`, `fetcher`,
`verifier`, `synthesizer`, `graph`, `llm`) and the CLI (`main.py`) were
written for this submission. Pure-logic components (URL canonicalization,
exact dedup, shingle near-dup, planner failure/fallback, fetch error
handling, ranker behavior, verifier checks) are covered by unit tests in
`tests/` that run without API keys; the opt-in embedding path, retrievers,
and LLM synthesis require network access / API keys and are exercised via
the end-to-end CLI run.
