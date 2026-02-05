# Use Python 3.11 slim image
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy source code
COPY src/ ./src/
COPY static/ ./static/

# Set environment variables
ENV PYTHONPATH=/app
ENV STATIC_DIR=/app/static

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "src.fintech_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]