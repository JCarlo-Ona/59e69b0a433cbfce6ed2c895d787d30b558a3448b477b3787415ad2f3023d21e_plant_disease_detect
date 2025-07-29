# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Copy dependency files first (for faster rebuilds)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/

# Create directories for mounted volumes
RUN mkdir -p data/raw data/processed models reports

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Command to run when container starts
CMD ["uv", "run", "python", "src/run_pipeline.py"]
