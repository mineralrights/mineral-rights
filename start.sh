#!/bin/bash

# Wait a moment for the container to fully initialize
sleep 5

# Start the FastAPI application
echo "🚀 Starting Mineral Rights API..."
uvicorn api.app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 28800 --timeout-graceful-shutdown 28800 --access-log
