# Use NVIDIA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set up model cache directory
VOLUME /app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV TORCH_HOME=/app/model_cache

# Copy application code
COPY . /app/
WORKDIR /app

# Run the application
CMD ["python3", "app.py"]

FROM python:3.10-slim as test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY . .
RUN pytest tests/unit tests/integration

FROM python:3.10-slim as production
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ /app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 