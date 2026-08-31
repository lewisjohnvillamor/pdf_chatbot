# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/lewisjohnvillamor/pdf_chatbot/security/advisories/new)
rather than opening a public issue.

Include the version or commit, reproduction steps, and what an attacker gains.
You can expect an initial response within a week.

## Scope

This is a self-hosted application. The following are in scope:

- Authentication bypass of the `APP_PASSWORD_HASH` gate
- Prompt injection that causes the assistant to ignore its grounding
  constraints in a way that misleads a user about what their documents say
- SQL injection or other injection in the pgvector store
- Path traversal or arbitrary file access via uploaded documents
- Secret disclosure through logs, error messages, or the UI

The following are **out of scope** because they are documented behaviour:

- An instance deployed with no `APP_PASSWORD_HASH` is unauthenticated by
  design. This is stated in the README; set the hash before exposing it.
- Document text is sent to your configured LLM provider. Use
  `EMBEDDING_PROVIDER=local` and a self-hosted `OPENAI_BASE_URL` if that is
  unacceptable for your material.
- The rate limiter is per-process and is not a defence against a distributed
  attacker. Enforce limits at your reverse proxy.
- The grounding check reduces hallucination; it does not eliminate it. Cited
  passages are always shown so a reader can verify.

## Deploying safely

The README's [Security notes](README.md#security-notes-for-a-public-deployment)
section is the deployment checklist. In short: set a password hash, terminate
TLS at a reverse proxy, keep Streamlit's XSRF protection on, and do not publish
the database port.
