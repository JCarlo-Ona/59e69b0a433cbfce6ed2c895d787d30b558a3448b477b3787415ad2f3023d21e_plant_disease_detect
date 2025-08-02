# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install --upgrade pip uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies (TensorFlow etc. should be listed in pyproject.toml)
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/

# Create directories for mounted volumes and outputs
RUN mkdir -p data/raw data/processed models reports

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Run the ML pipeline on container start
CMD ["uv", "run", "python", "src/run_pipeline.py"]
