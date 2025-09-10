#!/usr/bin/env python3
"""
Debug Deployment Issues
Check what's happening with the Document AI service initialization
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_environment():
    """Check environment variables"""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Check Google Cloud credentials
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    google_creds_b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
    doc_ai_creds = os.getenv("DOCUMENT_AI_CREDENTIALS_PATH")
    
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {'✅ Set' if google_creds else '❌ Not set'}")
    print(f"GOOGLE_CREDENTIALS_BASE64: {'✅ Set' if google_creds_b64 else '❌ Not set'}")
    print(f"DOCUMENT_AI_CREDENTIALS_PATH: {'✅ Set' if doc_ai_creds else '❌ Not set'}")
    
    # Check Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Not set'}")
    
    # Check Document AI endpoint
    doc_ai_endpoint = os.getenv("DOCUMENT_AI_ENDPOINT")
    print(f"DOCUMENT_AI_ENDPOINT: {'✅ Set' if doc_ai_endpoint else '❌ Not set'}")

def test_document_processor_initialization():
    """Test DocumentProcessor initialization"""
    print("\n🧪 Testing DocumentProcessor Initialization")
    print("=" * 50)
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        # Try to initialize with environment variables
        processor = DocumentProcessor()
        
        print("✅ DocumentProcessor initialized successfully")
        print(f"Document AI service available: {'✅ Yes' if processor.document_ai_service else '❌ No'}")
        
        if processor.document_ai_service:
            print("✅ Document AI service is properly initialized")
        else:
            print("❌ Document AI service is not available")
            print("   This could be due to missing credentials or initialization errors")
        
        return processor
        
    except Exception as e:
        print(f"❌ DocumentProcessor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_splitting_strategy():
    """Test splitting strategy handling"""
    print("\n🧪 Testing Splitting Strategy")
    print("=" * 50)
    
    processor = test_document_processor_initialization()
    if not processor:
        print("❌ Cannot test splitting strategy - processor not initialized")
        return
    
    try:
        # Test the splitting strategy
        print("Testing 'document_ai' strategy...")
        
        # This should not raise an error
        result = processor.split_pdf_by_deeds("data/multi-deed/pdfs/FRANCO.pdf", strategy="document_ai")
        print(f"✅ Strategy 'document_ai' accepted - found {len(result)} deeds")
        
    except Exception as e:
        print(f"❌ Strategy 'document_ai' failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Debug Deployment Issues")
    print("=" * 60)
    
    check_environment()
    test_splitting_strategy()
    
    print("\n" + "=" * 60)
    print("🔍 Debug complete")
