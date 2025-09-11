#!/bin/bash

# Ensure we're in the correct working directory
cd /app

# Verify the api directory exists
if [ ! -d "api" ]; then
    echo "❌ api directory not found in $(pwd)"
    ls -la
    exit 1
fi

# Start the FastAPI application
echo "🚀 Starting Mineral Rights API from $(pwd)..."
uvicorn api.app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 28800 --timeout-graceful-shutdown 28800 --access-log
