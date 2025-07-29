#!/usr/bin/env python3
"""
Mineral Rights Web App Launcher
==============================

Simple launcher script for the mineral rights classification web app.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import anthropic
        from document_classifier import DocumentProcessor
        print("✅ All dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    print("🏛️  Mineral Rights Classification Web App")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("document_classifier.py").exists():
        print("❌ Error: document_classifier.py not found")
        print("Please run this script from the mineral-rights directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("⚠️  Warning: ANTHROPIC_API_KEY not found in environment")
        print("The API key is currently hardcoded in document_classifier.py")
        print("For production use, please set the environment variable")
    
    print("\n📋 Starting web application...")
    print("🌐 The app will be available at: http://localhost:5001")
    print("📱 You can also access it from other devices on your network")
    print("🛑 Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down web server...")
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 