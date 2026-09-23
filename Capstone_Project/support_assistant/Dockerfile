# Use an official lightweight Python image
FROM python:3.10-slim

# Avoid buffering and writing bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1

WORKDIR /app

# Install system dependencies needed for compiling ChromaDB native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and corpus
COPY . .

# Pre-generate embeddings during container build for instant startup
RUN python ingest.py

# Expose port 7860
EXPOSE 7860

# Serve application via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]