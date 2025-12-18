#!/bin/bash

# Ensure we're in the correct working directory
cd /app

# Get Railway's PORT environment variable
PORT=${PORT:-8000}
echo "🚀 Starting Mineral Rights API on port $PORT..."

# Start the FastAPI application
uvicorn api.app_async:app --host 0.0.0.0 --port $PORT
