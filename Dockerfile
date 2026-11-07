FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for building C++ extensions and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install dependencies including llama-cpp-python and huggingface_hub
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir llama-cpp-python huggingface_hub

# Copy application source
COPY . .

# Ensure model directory exists and download model if not mounted
RUN python scripts/download_model.py

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=gguf
ENV LOCAL_MODEL_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf
ENV ENVIRONMENT=production

# Streamlit Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default entrypoint runs Streamlit UI
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
