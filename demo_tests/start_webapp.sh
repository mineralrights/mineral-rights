#!/bin/bash

echo "🏛️  Mineral Rights Document Analyzer - Web App"
echo "================================================"
echo ""
echo "Starting the web application..."
echo "Once started, open your browser and go to:"
echo "  http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server when you're done."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the web app
python run_webapp.py 