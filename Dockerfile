# Market DNA Engine | Docker Deployment
# Optimized for Algofest 2026 Submission

FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for numerical libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create cache directory for pre-downloaded data
RUN mkdir -p data/cache

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Healthcheck to verify the engine is alive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Entrypoint script to run both Streamlit and FastAPI if needed, 
# although Streamlit is the primary UI for judging.
CMD ["streamlit", "run", "0_Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
