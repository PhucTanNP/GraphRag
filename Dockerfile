FROM python:3.11-slim

# Install uv (fast Rust-based package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install system deps for neo4j driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy project files and install dependencies
COPY pyproject.toml .
COPY app/ ./app/
RUN uv sync

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
