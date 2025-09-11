#!/bin/bash

echo "=== Railway Container Startup Debugging ==="
echo "Current working directory: $(pwd)"
echo "Container contents:"
ls -la

# Ensure we're in the correct working directory
cd /app
echo "Changed to: $(pwd)"
echo "Contents of /app:"
ls -la

# Verify the api directory exists
if [ ! -d "api" ]; then
    echo "❌ api directory not found in $(pwd)"
    echo "Available directories:"
    ls -la
    exit 1
fi

echo "✅ api directory found"
echo "Contents of api directory:"
ls -la api/

# Test Python import before starting uvicorn
echo "Testing Python import..."
python3 -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python path: {sys.path}')
try:
    import api.app
    print('✅ api.app imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Python import test failed"
    exit 1
fi

# Start the FastAPI application
echo "🚀 Starting Mineral Rights API from $(pwd)..."
echo "Command: uvicorn api.app:app --host 0.0.0.0 --port 8000"
uvicorn api.app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 28800 --timeout-graceful-shutdown 28800 --access-log
