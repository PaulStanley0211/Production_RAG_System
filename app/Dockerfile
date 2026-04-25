# ============================================================
# Dockerfile — multi-stage build, non-root user
# Lives inside app/ as per the project template.
# Build context is the repo root; see docker-compose.yml.
# ============================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --prefix=/install .

# ------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app:/install/lib/python3.11/site-packages"

# Non-root user (production safety)
RUN useradd --create-home --shell /bin/bash --uid 1000 app

COPY --from=builder /install /install

WORKDIR /app
COPY --chown=app:app app/ ./app/
COPY --chown=app:app pipeline/ ./pipeline/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]