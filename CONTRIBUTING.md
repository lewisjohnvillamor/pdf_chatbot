# Contributing

Thanks for considering a contribution. This project has one organising
principle, and it shapes how changes are reviewed:

> **Claims about retrieval quality must come with numbers.**

If your change touches cleaning, chunking, retrieval, fusion, reranking or
prompting, run `make eval` before and after and put both tables in the pull
request. "This should be better" is not reviewable; a recall@k delta is.

## Getting set up

```bash
git clone https://github.com/lewisjohnvillamor/pdf_chatbot
cd pdf_chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env      # add an API key
make check                # lint + tests, must pass before you start
```

`make help` lists every target.

## Before opening a pull request

```bash
make fmt      # ruff format + autofix
make check    # lint + the full test suite
make eval     # retrieval metrics, if you touched the pipeline
```

CI runs the same three on Python 3.11 and 3.12, plus a Docker build.

## What a good change looks like

- **Tests come with it.** New behaviour needs a test that fails without the
  change. Bug fixes need a test that reproduces the bug first.
- **Errors are actionable.** An error message should tell the user what to do
  next, not just what went wrong. Compare `"Embedding failed"` against
  `"EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY. Set it, or switch to
  'local' or 'none'."`
- **Comments explain why, not what.** The code says what it does. A comment
  earns its place by explaining a decision, a constraint, or a trap.
- **Failure degrades, it doesn't cascade.** A failed grounding check returns
  "unchecked" and still shows the answer. A failed rerank falls back to fusion
  order. Nothing user-facing should die because an optional stage did.
- **Config is validated at load.** New settings go in `pdfchat/config.py` with
  a range check and an entry in `.env.example`.

## Project layout

| Path | What lives there |
| --- | --- |
| `pdfchat/` | All logic. Importable and testable without a browser or a network. |
| `app.py` | Streamlit UI. Presentation only — no retrieval or generation logic. |
| `tests/` | Unit and end-to-end tests. Real PDFs, faked chat model. |
| `evals/` | Labelled gold set and the retrieval harness. |
| `docs/` | Diagrams and screenshots. |

Provider SDK calls belong in `pdfchat/llm.py` and `pdfchat/embeddings.py` and
nowhere else. If you find yourself importing `anthropic` or `openai` in a
third file, add a method to the existing interface instead.

## Adding a provider

1. Implement the `ChatModel` protocol in `pdfchat/llm.py` (`complete`,
   `stream`, `last_usage`), translating SDK exceptions to `ProviderError`
   with messages a user can act on.
2. Register it in `build_chat_model`, add its models to
   `MODEL_PRICES_USD_PER_MTOK`, and document it in `.env.example`.
3. Add tests using a fake client — **no test may make a network call.**

## Reporting bugs

Include the output of `python -m pdfchat.cli check-config` (it prints no
secrets), what you expected, and what happened. If it concerns a specific PDF,
say whether it has a text layer: `pdftotext yourfile.pdf - | head`.

## Security

Please do not open a public issue for security problems. See
[SECURITY.md](SECURITY.md).

## Licence

Contributions are accepted under the [MIT Licence](LICENSE).
