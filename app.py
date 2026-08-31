"""Streamlit front end for the PDF study assistant.

The UI is deliberately thin: it collects input, calls :mod:`pdfchat.service`,
and renders results. All retrieval, generation and verification logic lives in
the package so it can be tested without a browser.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st
from dotenv import load_dotenv

from pdfchat.config import Settings, load_settings
from pdfchat.errors import PdfChatError
from pdfchat.export import (
    flashcards_to_markdown,
    flashcards_to_tsv,
    glossary_to_markdown,
    quiz_to_markdown,
    transcript_to_markdown,
)
from pdfchat.history import Turn, new_conversation_id, turn_from_answer
from pdfchat.logging_setup import configure_logging
from pdfchat.models import Answer, Usage
from pdfchat.prompts import EXPLAIN_LEVELS
from pdfchat.security import RateLimiter, verify_password
from pdfchat.service import ChatService, build_service
from pdfchat.study import gather_context, make_flashcards, make_glossary, make_quiz, make_summary

logger = logging.getLogger(__name__)


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """``3 passages`` / ``1 passage`` — small thing, but the UI shows it constantly."""
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count:,} {word}"


VERDICT_BADGES = {
    "grounded": ("✅", "Every claim checks out against the cited passages."),
    "partially_grounded": ("⚠️", "The main answer holds, but some details are not supported."),
    "ungrounded": ("🚩", "The cited passages do not support this answer. Treat it with caution."),
    "unchecked": ("", ""),
}


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    load_dotenv()
    settings = load_settings()
    configure_logging(settings.log_level)
    return settings


@st.cache_resource(show_spinner="Starting up…")
def get_service(_settings: Settings) -> ChatService:
    """One service graph per server process, shared across sessions.

    Streamlit reruns the script on every interaction, so building clients and
    connection pools here rather than in the script body is what keeps the app
    from opening a new pool per keystroke.
    """
    return build_service(_settings)


def init_state(settings: Settings) -> None:
    defaults = {
        "authenticated": not settings.auth_enabled,
        "conversation_id": new_conversation_id(),
        "answers": [],
        "session_usage": Usage(),
        "indexed": False,
        "index_report": None,
        "selected_docs": [],
        "level": "Intermediate",
        "pending_question": None,
        "summary": None,
        "glossary": None,
        "flashcards": None,
        "quiz": None,
        "quiz_answers": {},
        "quiz_submitted": False,
        "limiter": RateLimiter(max_events=settings.rate_limit_questions_per_hour),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ----------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------
def login_gate(settings: Settings) -> bool:
    """Show a password prompt when APP_PASSWORD_HASH is configured."""
    if st.session_state.authenticated:
        return True
    st.title("📚 PDF Study Assistant")
    st.caption("This instance is password protected.")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if verify_password(password, settings.app_password_hash or ""):
                st.session_state.authenticated = True
                logger.info("login_success")
                st.rerun()
            else:
                logger.warning("login_failed")
                st.error("Incorrect password.")
    return False


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def render_sidebar(service: ChatService) -> None:
    settings = service.settings
    with st.sidebar:
        st.header("Your documents")

        uploads = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Up to {settings.max_upload_mb} MB per file. Text-based PDFs only — "
            "run OCR on scans first.",
        )
        if st.button("Process documents", type="primary", disabled=not uploads):
            process_uploads(service, uploads)

        documents = service.documents()
        if documents:
            st.divider()
            st.subheader("Indexed")
            names = {name: doc_id for doc_id, name, _ in documents}
            for _doc_id, name, chunks in documents:
                st.caption(f"📄 {name} — {plural(chunks, 'passage')}")

            selected = st.multiselect(
                "Search within",
                options=list(names),
                default=[],
                help="Leave empty to search every document.",
            )
            st.session_state.selected_docs = [names[n] for n in selected]

        st.divider()
        st.session_state.level = st.select_slider(
            "Explanation level",
            options=list(EXPLAIN_LEVELS),
            value=st.session_state.level,
            help="Controls how much background the assistant assumes you have.",
        )

        render_usage_panel(service)

        st.divider()
        if st.button("Clear conversation"):
            st.session_state.answers = []
            st.session_state.conversation_id = new_conversation_id()
            st.rerun()
        if documents and st.button("Remove all documents"):
            service.reset()
            st.session_state.update(
                indexed=False, summary=None, glossary=None, flashcards=None, quiz=None
            )
            st.rerun()


def render_usage_panel(service: ChatService) -> None:
    settings = service.settings
    usage: Usage = st.session_state.session_usage
    with st.expander("Session cost & config", expanded=False):
        left, right = st.columns(2)
        left.metric("Tokens in", f"{usage.input_tokens:,}")
        right.metric("Tokens out", f"{usage.output_tokens:,}")
        if usage.cache_read_tokens:
            st.caption(f"Cache reads: {usage.cache_read_tokens:,} tokens (billed at ~10%)")
        st.metric("Estimated cost", f"${usage.cost_usd:.4f}")
        remaining = st.session_state.limiter.remaining
        if remaining >= 0:
            st.caption(f"Questions remaining this hour: {remaining}")
        st.caption(
            f"Chat: `{settings.chat_provider}/{settings.chat_model}` · "
            f"Embeddings: `{settings.embedding_provider}` · "
            f"Store: `{settings.vector_store}`"
        )


def process_uploads(service: ChatService, uploads: list) -> None:
    """Index uploaded files with a live progress bar."""
    progress = st.progress(0.0, text="Starting…")

    def report_progress(fraction: float, message: str) -> None:
        progress.progress(min(max(fraction, 0.0), 1.0), text=message)

    try:
        result = service.index(list(uploads), progress=report_progress)
    except PdfChatError as exc:
        progress.empty()
        st.error(exc.user_message)
        return
    except Exception:
        progress.empty()
        logger.exception("indexing_crashed")
        st.error("Indexing failed unexpectedly. Check the server logs for details.")
        return

    progress.empty()
    for problem in result.problems:
        st.warning(problem)
    for warning in result.warnings:
        st.info(warning)

    if not result.succeeded:
        st.error("Nothing could be indexed from those files.")
        return

    st.session_state.session_usage = st.session_state.session_usage.add(result.usage)
    st.session_state.update(
        indexed=True, index_report=result, summary=None, glossary=None, flashcards=None, quiz=None
    )
    source = "restored from cache" if result.from_cache else "indexed"
    st.success(
        f"{plural(len(result.documents), 'document')}, "
        f"{plural(result.page_count, 'page')}, "
        f"{plural(result.chunk_count, 'passage')} {source}."
    )
    st.rerun()


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------
def render_answer_extras(answer: Answer) -> None:
    """Show the evidence and the grounding verdict beneath an answer."""
    icon, explanation = VERDICT_BADGES.get(answer.verdict, ("", ""))
    if icon:
        note = f" {answer.verdict_note}" if answer.verdict_note else ""
        st.caption(f"{icon} {explanation}{note}")

    if answer.sources:
        with st.expander(f"📎 Sources ({plural(len(answer.sources), 'passage')} retrieved)"):
            cited = {c.marker for c in answer.citations}
            for index, scored in enumerate(answer.sources, start=1):
                chunk = scored.chunk
                mark = "✓ cited" if index in cited else "not cited"
                st.markdown(f"**[S{index}]** {chunk.locator} · _{mark}_")
                st.caption(
                    f"relevance {scored.score:.4f} · "
                    f"semantic {scored.dense_score:.3f} · keyword {scored.lexical_score:.2f}"
                )
                st.text(chunk.text[:900] + ("…" if len(chunk.text) > 900 else ""))
                st.divider()

    if answer.follow_ups:
        st.caption("Follow-up questions you could ask:")
        columns = st.columns(len(answer.follow_ups))
        for column, question in zip(columns, answer.follow_ups, strict=True):
            if column.button(question, key=f"followup-{hash(question)}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()


def render_chat(service: ChatService) -> None:
    st.subheader("Ask your documents")
    if not service.chunk_count():
        st.info("Upload a PDF in the sidebar to get started.")
        return

    for answer in st.session_state.answers:
        with st.chat_message("user"):
            st.markdown(answer.question)
        with st.chat_message("assistant"):
            st.markdown(answer.text)
            render_answer_extras(answer)

    question = st.chat_input("Ask a question about your documents…")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    if not question:
        return

    limiter: RateLimiter = st.session_state.limiter
    if not limiter.allow():
        st.error(
            f"Rate limit reached ({service.settings.rate_limit_questions_per_hour} questions "
            f"per hour). Try again in about {limiter.retry_after_seconds() // 60 + 1} minutes."
        )
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer: list[str] = []

        def on_token(piece: str) -> None:
            buffer.append(piece)
            placeholder.markdown("".join(buffer) + "▌")

        history = [(a.question, a.text) for a in st.session_state.answers]
        try:
            with st.spinner("Searching your documents…"):
                answer = service.pipeline.answer(
                    question,
                    collection=service.collection,
                    doc_ids=st.session_state.selected_docs or None,
                    level=st.session_state.level,
                    history=history,
                    on_token=on_token,
                )
        except PdfChatError as exc:
            placeholder.empty()
            st.error(exc.user_message)
            return
        except Exception:
            placeholder.empty()
            logger.exception("answer_crashed")
            st.error("The assistant hit an unexpected error. Check the server logs.")
            return

        placeholder.markdown(answer.text)
        if not answer.refused:
            answer.follow_ups = service.pipeline.suggest_follow_ups(answer)
        render_answer_extras(answer)

    st.session_state.answers.append(answer)
    st.session_state.session_usage = st.session_state.session_usage.add(answer.usage)
    conversation_id = st.session_state.conversation_id
    service.history.append(conversation_id, service.collection, Turn("user", question))
    service.history.append(conversation_id, service.collection, turn_from_answer(answer))
    st.rerun()


# ----------------------------------------------------------------------
# Study tools
# ----------------------------------------------------------------------
def study_context(service: ChatService, topic: str):
    sources, usage = gather_context(
        service.retriever,
        collection=service.collection,
        doc_ids=st.session_state.selected_docs or None,
        topic=topic,
    )
    st.session_state.session_usage = st.session_state.session_usage.add(usage)
    return sources


def run_study_tool(service: ChatService, label: str, topic: str, fn) -> None:
    """Shared wrapper: gather context, run ``fn``, bank the usage, report errors."""
    try:
        with st.spinner(f"Building {label}…"):
            sources = study_context(service, topic)
            if not sources:
                st.warning("No relevant passages found. Try a different topic.")
                return
            result, usage = fn(sources)
        st.session_state.session_usage = st.session_state.session_usage.add(usage)
        return result
    except PdfChatError as exc:
        st.error(exc.user_message)
    except Exception:
        logger.exception("study_tool_crashed", extra={"tool": label})
        st.error(f"Could not generate the {label}. Check the server logs.")
    return None


def render_study_tools(service: ChatService) -> None:
    if not service.chunk_count():
        st.info("Upload a PDF in the sidebar to unlock the study tools.")
        return

    settings = service.settings
    level = st.session_state.level
    topic = st.text_input(
        "Focus on a topic (optional)",
        placeholder="e.g. cellular respiration — leave blank to cover the whole document",
    )

    summary_tab, glossary_tab, cards_tab, quiz_tab = st.tabs(
        ["📝 Summary", "📖 Key terms", "🃏 Flashcards", "❓ Quiz"]
    )

    with summary_tab:
        if st.button("Generate summary", key="gen-summary"):
            st.session_state.summary = run_study_tool(
                service,
                "summary",
                topic,
                lambda sources: make_summary(service.model, sources, settings, level=level),
            )
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            st.download_button(
                "Download summary (Markdown)",
                st.session_state.summary,
                file_name="summary.md",
                mime="text/markdown",
            )

    with glossary_tab:
        count = st.slider("How many terms?", 5, 25, 12, key="glossary-count")
        if st.button("Extract key terms", key="gen-glossary"):
            st.session_state.glossary = run_study_tool(
                service,
                "glossary",
                topic,
                lambda sources: make_glossary(service.model, sources, settings, count=count),
            )
        if st.session_state.glossary:
            for term in st.session_state.glossary:
                st.markdown(f"**{term.term}** — {term.definition}")
                if term.source:
                    st.caption(f"Source: {term.source}")
            st.download_button(
                "Download glossary (Markdown)",
                glossary_to_markdown(st.session_state.glossary),
                file_name="glossary.md",
                mime="text/markdown",
            )

    with cards_tab:
        count = st.slider("How many cards?", 5, 30, 10, key="cards-count")
        if st.button("Generate flashcards", key="gen-cards"):
            st.session_state.flashcards = run_study_tool(
                service,
                "flashcards",
                topic,
                lambda sources: make_flashcards(
                    service.model, sources, settings, count=count, level=level
                ),
            )
        cards = st.session_state.flashcards
        if cards:
            for index, card in enumerate(cards, start=1):
                with st.expander(f"{index}. {card.front}"):
                    st.markdown(card.back)
                    if card.source:
                        st.caption(f"Source: {card.source}")
            left, right = st.columns(2)
            left.download_button(
                "Download for Anki (TSV)",
                flashcards_to_tsv(cards),
                file_name="flashcards.tsv",
                mime="text/tab-separated-values",
            )
            right.download_button(
                "Download as Markdown",
                flashcards_to_markdown(cards),
                file_name="flashcards.md",
                mime="text/markdown",
            )

    with quiz_tab:
        count = st.slider("How many questions?", 3, 15, 5, key="quiz-count")
        if st.button("Generate quiz", key="gen-quiz"):
            st.session_state.quiz = run_study_tool(
                service,
                "quiz",
                topic,
                lambda sources: make_quiz(
                    service.model, sources, settings, count=count, level=level
                ),
            )
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
        render_quiz()


def render_quiz() -> None:
    questions = st.session_state.quiz
    if not questions:
        return
    with st.form("quiz-form"):
        for index, question in enumerate(questions):
            st.markdown(f"**{index + 1}. {question.question}**")
            st.session_state.quiz_answers[index] = st.radio(
                "Choose one",
                options=list(range(len(question.options))),
                format_func=lambda i, q=question: q.options[i],
                key=f"quiz-{index}",
                index=None,
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Check answers", type="primary")
    if submitted:
        st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        correct = 0
        for index, question in enumerate(questions):
            chosen = st.session_state.quiz_answers.get(index)
            if chosen == question.answer_index:
                correct += 1
                st.success(f"**{index + 1}. Correct** — {question.answer_text}")
            else:
                picked = question.options[chosen] if chosen is not None else "no answer"
                st.error(
                    f"**{index + 1}. Not quite** — you chose _{picked}_; "
                    f"the answer is _{question.answer_text}_"
                )
            if question.explanation:
                st.caption(question.explanation)
            if question.source:
                st.caption(f"Source: {question.source}")
        st.metric("Score", f"{correct} / {len(questions)}")
        st.download_button(
            "Download quiz (Markdown)",
            quiz_to_markdown(questions),
            file_name="quiz.md",
            mime="text/markdown",
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="PDF Study Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    try:
        settings = get_settings()
    except PdfChatError as exc:
        st.error(f"Configuration error: {exc.user_message}")
        st.stop()

    init_state(settings)
    if not login_gate(settings):
        return

    missing = settings.missing_credentials()
    if missing:
        st.error(
            "Missing required credentials: "
            + ", ".join(missing)
            + ". Set them in your environment or .env file, then restart."
        )
        st.stop()

    try:
        service = get_service(settings)
    except PdfChatError as exc:
        st.error(exc.user_message)
        st.stop()

    st.title("📚 PDF Study Assistant")
    st.caption(
        "Answers are drawn only from your uploaded documents, cited to the page, "
        "and checked against the sources before you see them."
    )

    render_sidebar(service)

    chat_tab, study_tab, export_tab = st.tabs(["💬 Chat", "🎓 Study tools", "⬇️ Export"])
    with chat_tab:
        render_chat(service)
    with study_tab:
        render_study_tools(service)
    with export_tab:
        render_export()


def render_export() -> None:
    answers: list[Answer] = st.session_state.answers
    if not answers:
        st.info("Ask a few questions first — your transcript will appear here.")
        return
    transcript = transcript_to_markdown(answers)
    st.download_button(
        "Download transcript (Markdown)",
        transcript,
        file_name="study-session.md",
        mime="text/markdown",
        type="primary",
    )
    st.markdown("#### Preview")
    st.markdown(transcript)


if __name__ == "__main__":
    main()
