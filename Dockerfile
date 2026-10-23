# ==============================================================================
# Cloud-Based RAG Document Assistant - Cloud Run Dockerfile
# Production-ready, secure, lightweight multi-stage container
# ==============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

WORKDIR /app

# Install system dependencies (build-essential for chromadb hnswlib compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create non-privileged user and persist directory permissions
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Container user" appuser && \
    mkdir -p /app/chroma_db && \
    chown -R appuser:appuser /app

USER appuser

# Healthcheck for Cloud Run / Kubernetes container probes
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

EXPOSE 8080

# Execute Streamlit server
ENTRYPOINT ["streamlit", "run", "app.py"]
