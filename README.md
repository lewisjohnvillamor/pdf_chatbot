# 📚 PDF Study Assistant

A self-hosted study assistant that answers questions **only** from your own PDFs,
cites every claim back to the page it came from, and verifies its own answers
against the sources before you see them.

Built to be run for real: pluggable LLM providers, a Postgres/pgvector store,
a document-processing pipeline that repairs the mess PDF extraction produces,
structured logging, auth, rate limiting, and a test suite.

---

## Why this isn't just an API wrapper

Most "chat with your PDF" apps concatenate `extract_text()` output, split it
every 1000 characters, and hand the result to an LLM. That fails in specific,
predictable ways. This project addresses each one:

| Problem with the naive approach | What this does instead |
| --- | --- |
| Hyphenated words split across lines (`informa-\ntion`) become junk tokens | De-hyphenation and soft-wrap repair rejoin them (`pdfchat/cleaning.py`) |
| Running headers and page numbers repeat on every chunk, polluting retrieval | Cross-page repetition analysis strips furniture while protecting body text |
| Fixed-size splits cut sentences in half and lose the page number | Structure-aware chunking packs whole paragraphs and records the page span |
| Repeated boilerplate gets embedded once per page | SimHash near-duplicate collapsing removes it before embedding |
| Pure vector search misses exact terms (`Article 7(b)`, `NADPH`) | Hybrid BM25 + dense retrieval fused with Reciprocal Rank Fusion |
| Nobody knows whether any of it actually helps | A labelled gold set and a retrieval eval harness — see [Measuring retrieval](#measuring-retrieval) |
| Top-k returns six near-copies of one paragraph | Maximal Marginal Relevance diversifies the context |
| The model cites a page that doesn't support the claim | A second verification pass checks each claim against the excerpts |
| The model invents `[S7]` when only 5 sources exist | Fabricated citation markers are detected and stripped |
| Re-uploading the same PDF re-pays the whole embedding bill | Content-hash fingerprinting caches the built index |
| The corpus dies with the process | Optional Postgres + pgvector persistence, shared across replicas |

---

## Measuring retrieval

Generation quality is downstream of retrieval: if the passage containing the
answer never reaches the prompt, no model and no prompt can recover it. So the
project ships a labelled gold set (22 scored questions plus a negative control)
and a harness that runs the **whole production path** — real PDFs through
ingest, clean, chunk, embed, index — and reports recall@k, MRR and nDCG@k.

```bash
make eval        # offline, no API cost
make eval-real   # uses your configured embedding provider
```

### What the offline run currently reports

```
Corpus: 4 documents, 22 passages at chunk_size=400
Gold set: 22 scored questions (+1 negative control), k=5

configuration               recall@k     MRR   nDCG@k  misses   cover
---------------------------------------------------------------------
BM25 only                      0.864   0.841    0.828       3   23%
hybrid w=0.25                  0.864   0.841    0.828       3   23%
hybrid w=0.5                   0.864   0.841    0.828       3   23%
hybrid w=0.75                  0.864   0.841    0.828       3   23%
dense only                     0.864   0.841    0.828       3   23%
```

**Every configuration scores identically, and the harness says so.** The
offline embedder is a hashed bag-of-words — the same signal BM25 already uses —
so the two rankers return the same order and RRF has nothing to fuse. This run
*cannot* tell you what to set `HYBRID_DENSE_WEIGHT` to, and the tool refuses to
crown a winner rather than manufacturing one. Run `make eval-real` against a
real embedding model to get an answer you can act on.

That is the honest current state: **the hybrid retrieval is implemented and
tested, but its benefit over BM25 alone is not yet demonstrated on this
corpus.** The eval exists so that claim can be settled with numbers instead of
argument.

### Guards against fooling yourself

The harness refuses to produce misleading numbers:

- **Degenerate corpus** — if the corpus has fewer than `2 × k` passages it
  exits with an error, because every configuration would trivially score 1.0.
- **Coverage confound** — a `cover` column reports what fraction of the corpus
  top-k returns, and rows above 25% are flagged `!`. Chunking at 700 characters
  scores a perfect 1.000 recall on this corpus, but does it at 42% coverage —
  it "wins" by returning most of the corpus, not by ranking well. Flagged, not
  celebrated.
- **Negative controls** — a question the corpus cannot answer is excluded from
  ranking metrics. Retrieval always returns its nearest passages; refusing is
  the generator's job, tested separately in `tests/test_rag.py`.

### The three questions retrieval currently misses

Kept visible rather than hidden, because they describe the real failure modes:

| Question | Wanted | Why it's hard |
| --- | --- | --- |
| "What does Article 7(c) allow?" | `extension of up to 30 days` | Rare identifier with near-identical sibling clauses 7(a)/7(b) |
| "What is the maximum fine in one reporting period?" | `capped at 3000 euros` | Paraphrase — "maximum fine" never appears as those words |
| "Why could factories move away from rivers?" | `steam engine` | Causal question; the answer is stated indirectly |

---

## Features

**Grounded answering**
- Every factual claim carries a `[S1]`-style marker resolving to a file and page.
- An explicit "The documents don't cover this." path — no filling gaps with
  general knowledge.
- A grounding verdict (✅ / ⚠️ / 🚩) on every answer, from a second checking pass.
- The retrieved passages are always shown, marked cited or not cited, with
  their relevance, semantic and keyword scores.

**Study tools** — all grounded in the same retrieved passages
- Cited study summaries
- Key-term glossaries
- Flashcards, exportable as Anki-ready TSV
- Multiple-choice quizzes with scoring and per-question explanations

**Learner controls**
- Explanation level: Beginner → Intermediate → Expert
- Per-document scoping ("search only in `lecture-3.pdf`")
- Streaming answers, suggested follow-up questions
- Markdown transcript export

**Operations**
- Per-session token and USD cost tracking, with prompt-cache accounting
- Structured JSON logging on stdout
- PBKDF2 password gate and per-session rate limiting
- Upload size/page guards and scanned-PDF (no text layer) detection
- Non-root, read-only container; health checks; CI

---

## Quick start

### Local

```bash
git clone <your-fork> && cd pdf_chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # add ANTHROPIC_API_KEY (or OPENAI_API_KEY)
make config               # validate before you start
make run                  # http://localhost:8501
```

### Self-hosted with Docker (recommended)

```bash
cp .env.example .env
# In .env: set your API key, VECTOR_STORE=postgres, and an APP_PASSWORD_HASH
python -m pdfchat.cli hash-password     # prints the APP_PASSWORD_HASH line

docker compose up -d --build
```

That brings up the app plus a `pgvector/pgvector:pg16` database. The database
port is not published — only the app reaches it over the internal network.
Chunks, embeddings and conversations survive restarts in named volumes.

---

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example) for the
annotated list. The settings that change behaviour most:

| Variable | Default | Notes |
| --- | --- | --- |
| `CHAT_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `CHAT_MODEL` | `claude-opus-5` | `gpt-4.1-mini` when provider is `openai` |
| `EMBEDDING_PROVIDER` | `openai` | `openai`, `local` (on-device), or `none` |
| `VECTOR_STORE` | `memory` | `memory` (cached NumPy) or `postgres` (pgvector) |
| `HYBRID_DENSE_WEIGHT` | `0.5` | 0.0 = keyword only, 1.0 = semantic only |
| `MMR_LAMBDA` | `0.6` | Lower = more diverse retrieved passages |
| `ENABLE_SELF_CHECK` | `true` | The grounding verification pass |
| `APP_PASSWORD_HASH` | *(unset)* | Set this before exposing the app |
| `RATE_LIMIT_QUESTIONS_PER_HOUR` | `120` | Per session |

### Running with no embedding provider

Set `EMBEDDING_PROVIDER=none` and the app runs on BM25 keyword retrieval alone
— no embedding API, no cost, fully offline. Recall on paraphrased questions
drops, but citations and grounding still work exactly the same.

### Running fully offline

```bash
pip install -r requirements-local.txt   # adds sentence-transformers
# .env: EMBEDDING_PROVIDER=local
```

---

## Architecture

```
Upload → ingest → cleaning → chunking → embeddings → vector store
                                                          │
Question ──────────────────────────────────────────────┐  │
                                                       ▼  ▼
                              BM25 ─┐            dense search
                                    ├─ RRF fusion → MMR → top-k passages
                                    │                          │
                                    └──────────────────────────┤
                                                               ▼
                                          prompt (sources + question)
                                                               │
                                                    LLM (streamed)
                                                               │
                                     citation resolution + grounding check
                                                               │
                                                          Answer + sources
```

| Module | Responsibility |
| --- | --- |
| `pdfchat/config.py` | Environment-driven, validated settings — the only place that reads `os.environ` |
| `pdfchat/ingest.py` | PDF reading, size/page/encryption guards, scanned-PDF detection |
| `pdfchat/cleaning.py` | Unicode/ligature normalization, de-hyphenation, header & footer removal |
| `pdfchat/chunking.py` | Structure-aware chunking, page provenance, near-duplicate collapsing |
| `pdfchat/lexical.py` | BM25 Okapi implementation |
| `pdfchat/embeddings.py` | OpenAI / local / null embedding backends |
| `pdfchat/stores/` | `memory` (NumPy + disk cache) and `postgres` (pgvector) stores |
| `pdfchat/retrieval.py` | Hybrid search: RRF fusion and MMR diversification |
| `pdfchat/llm.py` | Anthropic and OpenAI chat backends, streaming, cost accounting |
| `pdfchat/rag.py` | The answer pipeline |
| `pdfchat/grounding.py` | Post-generation claim verification |
| `pdfchat/citations.py` | Marker resolution and fabrication stripping |
| `pdfchat/study.py` | Summaries, glossaries, flashcards, quizzes |
| `pdfchat/evaluation.py` | Ranking metrics, config sweeps, confound detection |
| `pdfchat/service.py` | Composition root |
| `app.py` | Streamlit UI — presentation only |

---

## Development

```bash
pip install -r requirements-dev.txt
make check        # lint + tests
make test-cov     # coverage report
```

The suite covers the pipeline end to end: real PDFs are generated with
reportlab, then ingested, cleaned, chunked, embedded, indexed, retrieved and
cited. Only the chat provider is faked.

```
make help         # all available targets
```

---

## Security notes for a public deployment

1. **Set `APP_PASSWORD_HASH`.** Without it, anyone who reaches the URL can spend
   your API budget. `make password` generates the value.
2. **Terminate TLS at a reverse proxy** (nginx, Caddy, Traefik) and forward the
   `X-Forwarded-*` headers. Leave Streamlit's XSRF protection enabled.
3. **Uploaded PDFs are processed in memory** and never written to disk; only
   extracted text and embeddings are persisted.
4. **Document text is sent to your configured LLM provider.** For material that
   cannot leave your infrastructure, use `EMBEDDING_PROVIDER=local` and point
   `CHAT_PROVIDER` at a self-hosted OpenAI-compatible endpoint via
   `OPENAI_BASE_URL`.
5. `showErrorDetails = false` in `.streamlit/config.toml` keeps stack traces out
   of the browser; they go to the structured logs instead.

---

## Limitations

- **Scanned PDFs are rejected** rather than silently returning nothing. Run
  `ocrmypdf in.pdf out.pdf` first.
- **Tables and multi-column layouts** extract imperfectly — a limitation of
  text-layer extraction, not of the pipeline above it.
- The grounding check reduces hallucination; it does not eliminate it. The
  cited passages are always shown so you can confirm for yourself.
- The rate limiter is per-process. Across multiple replicas, enforce limits at
  the reverse proxy.
- **Hybrid retrieval is unproven on the shipped corpus.** See
  [Measuring retrieval](#measuring-retrieval). The mechanism is implemented and
  unit-tested; whether it beats BM25 alone on *your* documents is a question
  `make eval-real` answers, and the answer may be no.
- There is no cross-encoder reranker. That is usually the next largest win
  after hybrid retrieval, and the eval harness is the right place to justify
  adding one.
- Token counts in the cost panel come from the provider; the `chunk_size`
  budget uses a ~4-characters-per-token estimate, which is approximate.

## Licence

MIT — see [LICENSE](LICENSE).
