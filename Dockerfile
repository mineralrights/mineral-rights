# Use a small, stable Python image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools only if a wheel needs compiling
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (API only - main requirements.txt has conda-specific packages)
COPY api/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (API + classifier)
COPY api ./api
COPY src ./src

# Expose the default FastAPI port
EXPOSE 8000

# Start the service with extended timeouts for very long-running processes (8+ hours)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "28800", "--timeout-graceful-shutdown", "28800"] 