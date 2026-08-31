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
| Top-k returns six near-copies of one paragraph | Maximal Marginal Relevance diversifies the context |
| The model cites a page that doesn't support the claim | A second verification pass checks each claim against the excerpts |
| The model invents `[S7]` when only 5 sources exist | Fabricated citation markers are detected and stripped |
| Re-uploading the same PDF re-pays the whole embedding bill | Content-hash fingerprinting caches the built index |
| The corpus dies with the process | Optional Postgres + pgvector persistence, shared across replicas |

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

## Licence

MIT — see [LICENSE](LICENSE).
