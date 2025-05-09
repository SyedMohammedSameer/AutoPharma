FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory for GGUF files
RUN mkdir -p models

# Copy application code
COPY . .

# Download GGUF model files if needed
RUN mkdir -p /app/models && \
    if [ ! -f /app/models/phi-3-mini-4k-instruct.Q4_K_M.gguf ]; then \
        echo "GGUF model file not found. You may need to manually add it to the models directory."; \
    fi

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]