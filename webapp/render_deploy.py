#!/usr/bin/env python3
"""
Render deployment helper script
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check current dependency versions"""
    print("🔍 Checking current dependency versions...")
    
    try:
        import anthropic
        print(f"✅ anthropic: {anthropic.__version__}")
    except ImportError as e:
        print(f"❌ anthropic not found: {e}")
    
    try:
        import flask
        print(f"✅ flask: {flask.__version__}")
    except ImportError as e:
        print(f"❌ flask not found: {e}")
    
    try:
        import fitz
        print(f"✅ PyMuPDF: imported successfully")
    except ImportError as e:
        print(f"❌ PyMuPDF not found: {e}")

def force_reinstall():
    """Force reinstall of dependencies"""
    print("🔄 Force reinstalling dependencies...")
    
    commands = [
        "pip uninstall -y anthropic",
        "pip install anthropic==0.25.0",
        "pip install --force-reinstall -r requirements.txt"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {result.stderr}")
        else:
            print(f"✅ Command succeeded")

def main():
    """Main deployment check"""
    print("🚀 Render Deployment Helper")
    print("=" * 40)
    
    # Check environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"✅ ANTHROPIC_API_KEY found (length: {len(api_key)})")
    else:
        print("❌ ANTHROPIC_API_KEY not found!")
    
    # Check dependencies
    check_dependencies()
    
    # Test basic import
    try:
        from document_classifier import DocumentProcessor
        print("✅ DocumentProcessor import successful")
        
        # Try to initialize (this will fail without API key but shows the error)
        try:
            processor = DocumentProcessor()
            print("✅ DocumentProcessor initialization successful")
        except Exception as e:
            print(f"⚠️  DocumentProcessor initialization failed: {e}")
            
    except Exception as e:
        print(f"❌ DocumentProcessor import failed: {e}")

if __name__ == "__main__":
    main() 