#!/bin/bash

echo "🏛️  Mineral Rights Classification - Streamlit Web App"
echo "=" * 60

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found"
    echo "Please run this script from the mineral-rights directory"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import streamlit, anthropic, fitz, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not found in environment"
    echo "You can set it with: export ANTHROPIC_API_KEY='your-key-here'"
    echo "Or create .streamlit/secrets.toml with your API key"
fi

echo ""
echo "🚀 Starting Streamlit application..."
echo "🌐 The app will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false 