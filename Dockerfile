# syntax=docker/dockerfile:1

# ---- builder -------------------------------------------------------------
# Dependencies are installed into a virtualenv that is copied into the runtime
# image, so build tooling never ships to production.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- runtime -------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CACHE_DIR=/data/cache

# Run unprivileged: a compromised app process must not own its own filesystem.
RUN groupadd --system --gid 1001 pdfchat \
    && useradd --system --uid 1001 --gid pdfchat --create-home pdfchat

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=pdfchat:pdfchat pdfchat/ ./pdfchat/
COPY --chown=pdfchat:pdfchat app.py ./
COPY --chown=pdfchat:pdfchat .streamlit/ ./.streamlit/

RUN mkdir -p /data/cache && chown -R pdfchat:pdfchat /data
VOLUME ["/data"]

USER pdfchat
EXPOSE 8501

# Streamlit's own health endpoint — used by compose and by orchestrators.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
