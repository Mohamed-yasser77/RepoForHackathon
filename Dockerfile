# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Smart-Support Milestone 2 (Multi-stage Optimized)
# ─────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build dependencies (compilers, etc. needed for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies into the venv without keeping the cache
# (Optimisation: Use CPU-only PyTorch to save ~2GB of Docker image size)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# ─── Stage 2: Final Runtime ──────────────────────────────────────────────────
FROM python:3.11-slim

# Security: standard practice to run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

WORKDIR /app

# Copy the pre-built dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY . .

# Change ownership to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Default command (overridden per service in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
