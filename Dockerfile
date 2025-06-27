FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and UV
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  curl \
  && rm -rf /var/lib/apt/lists/* \
  && curl -LsSf https://astral.sh/uv/install.sh | sh \
  && mv /root/.cargo/bin/uv /usr/local/bin/

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install dependencies with UV
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/transcripts

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Expose port for FastAPI server
EXPOSE 8000

# Default command (can be overridden)
CMD ["uv", "run", "python", "main.py", "status"]
