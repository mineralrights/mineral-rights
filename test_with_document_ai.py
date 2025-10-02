#!/usr/bin/env python3
"""
Test script that includes Document AI setup for proper multi-deed processing
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mineral_rights.document_classifier import DocumentProcessor

def setup_document_ai():
    """Set up Document AI environment variables"""
    
    print("🔧 Setting up Document AI...")
    
    # Set Document AI endpoint
    document_ai_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf:process"
    os.environ["DOCUMENT_AI_ENDPOINT"] = document_ai_endpoint
    print(f"✅ Set DOCUMENT_AI_ENDPOINT: {document_ai_endpoint}")
    
    # Check if we have Google credentials
    if os.path.exists("service-account.json"):
        print("✅ Found service-account.json file")
        return True
    else:
        print("⚠️ No service-account.json found. Document AI will use default credentials.")
        print("   Make sure you've run: gcloud auth application-default login")
        return True

def test_with_document_ai(pdf_path: str):
    """Test the complete pipeline with Document AI enabled"""
    
    print(f"🧪 Testing with Document AI: {pdf_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        return False
    
    try:
        # Initialize processor with Document AI
        print("🔧 Initializing document processor with Document AI...")
        
        # Set up Document AI
        setup_document_ai()
        
        # Initialize processor (use default credentials, not the corrupted service-account.json)
        processor = DocumentProcessor(
            api_key=api_key,
            document_ai_endpoint=os.getenv("DOCUMENT_AI_ENDPOINT"),
            document_ai_credentials=None  # Use default Google Cloud credentials
        )
        print("✅ Processor initialized successfully")
        
        # Test multi-deed processing with Document AI
        print("\n📚 Testing multi-deed processing with Document AI...")
        start_time = time.time()
        
        multi_result = processor.process_multi_deed_document(
            pdf_path,
            strategy="document_ai"  # Force Document AI usage
        )
        
        multi_time = time.time() - start_time
        print(f"✅ Multi-deed processing completed in {multi_time:.1f}s")
        print(f"   Total deeds processed: {len(multi_result)}")
        
        # Show results for each deed
        for i, deed_result in enumerate(multi_result):
            print(f"   Deed {i+1}: {deed_result['classification']} (confidence: {deed_result['confidence']:.3f})")
            if 'error' in deed_result:
                print(f"     Error: {deed_result['error']}")
            if 'deed_boundary_info' in deed_result:
                boundary = deed_result['deed_boundary_info']
                print(f"     Pages: {boundary['page_range']} (confidence: {boundary['confidence']:.3f})")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Multi-deed time: {multi_time:.1f}s")
        print(f"   Total deeds found: {len(multi_result)}")
        print(f"   Document AI used: {any('deed_boundary_info' in deed for deed in multi_result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main testing function"""
    
    print("🚀 Mineral Rights Processor - Document AI Testing")
    print("=" * 50)
    
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Look for test PDFs in common locations
        test_paths = [
            "data/reservs/Washington DB 405_547.pdf",
            "data/no-reservs/sample.pdf",
            "test_small.pdf"
        ]
        
        pdf_path = None
        for path in test_paths:
            if os.path.exists(path):
                pdf_path = path
                break
        
        if not pdf_path:
            print("❌ No test PDF found. Please provide a PDF path:")
            print("   python test_with_document_ai.py /path/to/your/pdf")
            return
    
    print(f"📁 Using PDF: {pdf_path}")
    
    # Check file size
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    print(f"📊 File size: {file_size:.1f} MB")
    
    if file_size > 100:
        print("⚠️  Large file detected. This may take a while...")
    
    # Run tests
    success = test_with_document_ai(pdf_path)
    
    if success:
        print("\n✅ Document AI testing completed successfully!")
        print("🚀 Ready for Cloud Run deployment with Document AI!")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        print("\n🔧 To fix Document AI issues:")
        print("   1. Run: gcloud auth application-default login")
        print("   2. Or place service-account.json in the project root")
        sys.exit(1)

if __name__ == "__main__":
    main()
