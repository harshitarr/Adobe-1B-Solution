# Round 1B Persona-Driven Document Intelligence
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV NLTK_DATA=/app/models/nlp/nltk_data

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU with COMPATIBLE versions
RUN pip install torch==2.1.1+cpu torchvision==0.16.1+cpu torchaudio==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file (without PyTorch lines)
COPY requirements.txt .

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/logs /app/data/cache /app/models

# Download and setup models (with proper Python path)
RUN PYTHONPATH=/app python scripts/setup.py

# Create verification script for models
RUN echo 'import sys; from sentence_transformers import SentenceTransformer; model = SentenceTransformer("all-MiniLM-L6-v2"); print("Model loaded successfully"); sys.exit(0)' > /tmp/verify_model.py
RUN python /tmp/verify_model.py

# Verify spaCy model
RUN echo 'import spacy; nlp = spacy.load("en_core_web_sm"); print("spaCy loaded successfully")' > /tmp/verify_spacy.py
RUN python /tmp/verify_spacy.py

# Create symlinks for NLTK data to standard locations (backup fallback)
RUN if [ -d "/app/models/nlp/nltk_data" ]; then \
        mkdir -p /root/nltk_data && \
        ln -sf /app/models/nlp/nltk_data/* /root/nltk_data/ 2>/dev/null || true; \
    fi

# Create additional NLTK data fallback locations
RUN mkdir -p /usr/local/nltk_data /usr/local/share/nltk_data && \
    if [ -d "/app/models/nlp/nltk_data" ]; then \
        ln -sf /app/models/nlp/nltk_data/* /usr/local/nltk_data/ 2>/dev/null || true && \
        ln -sf /app/models/nlp/nltk_data/* /usr/local/share/nltk_data/ 2>/dev/null || true; \
    fi

# Copy and set permissions for entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Clean up temporary files
RUN rm -f /tmp/verify_*.py

# Expose volume mount points
VOLUME ["/app/input", "/app/output"]

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
