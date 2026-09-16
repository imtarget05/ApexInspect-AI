FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (SDC best practice)
RUN groupadd -r apexinspect && useradd -r -g apexinspect -d /app -s /sbin/nologin apexinspect

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY --chown=apexinspect:apexinspect . .

# Switch to non-root user
USER apexinspect

# Hugging Face Spaces / Azure Container Apps default port
EXPOSE 7860

# Healthcheck: branch by APP_MODE (api -> /health, ui -> Streamlit health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c "if [ \"$APP_MODE\" = \"api\" ]; then curl -fsS http://localhost:${PORT:-7860}/health || exit 1; else curl -fsS http://localhost:${PORT:-7860}/_stcore/health || exit 1; fi"

# Streamlit defaults for container environment
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Multi-Space entrypoint: APP_MODE=api runs the FastAPI gateway,
# anything else (default) runs the Streamlit Operator Console.
# Both bind to 0.0.0.0 on the single HF Space public port.
CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"api\" ]; then exec uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}; else exec streamlit run src/ui/app.py --server.port=${PORT:-7860} --server.address=0.0.0.0; fi"]


