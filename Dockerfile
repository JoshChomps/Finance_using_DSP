# --- Build Stage ---
FROM python:3.10-slim as builder

WORKDIR /app

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy only requirements to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Final Stage ---
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for scipy/numpy if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Ensure the data cache directory exists
RUN mkdir -p /app/data/cache

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FIN_DATA_CACHE=/app/data/cache

# Labels
LABEL org.opencontainers.image.title="FinSignal Suite"
LABEL org.opencontainers.image.description="Multi-Resolution Financial Signal Processing Engine"

# We don't specify CMD here; Docker Compose will handle the entrypoints
