# Multi-stage Dockerfile for mmWave Human Identification Platform
# Optimized for PyTorch ML training with web GUI

# Base stage with Python and PyTorch
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/results/checkpoints \
    /app/results/tensorboard \
    /app/results/reports

# Copy requirements files
COPY requirements.txt /app/
COPY requirements-gui.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-gui.txt

# Copy application code
COPY src/ /app/src/
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY config.yaml /app/

# Expose port for web interface
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=backend/app.py
ENV PORT=5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the Flask application
CMD ["python", "backend/app.py"]
