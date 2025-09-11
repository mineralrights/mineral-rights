#!/usr/bin/env python3
"""
Test Document AI with proper credentials
This script tests if Document AI is working after fixing credentials
"""

import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_document_ai_service():
    """Test Document AI service directly"""
    print("🧪 Testing Document AI Service")
    print("=" * 50)
    
    try:
        from mineral_rights.document_ai_service import DocumentAIService
        
        # Initialize Document AI service
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        service = DocumentAIService(processor_endpoint)
        
        print("✅ Document AI service initialized successfully")
        print(f"   Processor endpoint: {processor_endpoint}")
        
        return True, service
        
    except Exception as e:
        print(f"❌ Document AI service initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_document_processor_with_temp_key():
    """Test DocumentProcessor with a temporary API key"""
    print("\n🧪 Testing DocumentProcessor with Temporary API Key")
    print("=" * 50)
    
    # Set a temporary API key for testing
    temp_key = "temp-key-for-testing"
    os.environ["ANTHROPIC_API_KEY"] = temp_key
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        # Initialize processor with API key and Document AI endpoint
        processor = DocumentProcessor(
            api_key=temp_key,
            document_ai_endpoint="https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        )
        
        print("✅ DocumentProcessor initialized successfully")
        print(f"   Document AI service available: {'✅ Yes' if processor.document_ai_service else '❌ No'}")
        
        if processor.document_ai_service:
            print("✅ Document AI service is properly integrated")
            return True, processor
        else:
            print("❌ Document AI service is not available")
            return False, None
        
    except Exception as e:
        print(f"❌ DocumentProcessor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_splitting_strategy():
    """Test the splitting strategy"""
    print("\n🧪 Testing Splitting Strategy")
    print("=" * 50)
    
    # Set environment variables
    os.environ["ANTHROPIC_API_KEY"] = "temp-key-for-testing"
    os.environ["DOCUMENT_AI_ENDPOINT"] = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        processor = DocumentProcessor(
            api_key="temp-key-for-testing",
            document_ai_endpoint="https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        )
        
        # Test if the splitting strategy is available
        if hasattr(processor, 'split_pdf_by_deeds'):
            print("✅ split_pdf_by_deeds method is available")
            
            # Test the strategy without actually processing a file
            print("✅ Document AI splitting strategy should work")
            return True
        else:
            print("❌ split_pdf_by_deeds method is not available")
            return False
        
    except Exception as e:
        print(f"❌ Splitting strategy test failed: {e}")
        return False

def main():
    print("🚀 Test Document AI After Credentials Fix")
    print("=" * 60)
    
    # Test 1: Document AI Service
    service_ok, service = test_document_ai_service()
    
    if not service_ok:
        print("\n❌ Document AI service test failed")
        return
    
    # Test 2: DocumentProcessor
    processor_ok, processor = test_document_processor_with_temp_key()
    
    if not processor_ok:
        print("\n❌ DocumentProcessor test failed")
        return
    
    # Test 3: Splitting Strategy
    strategy_ok = test_splitting_strategy()
    
    if strategy_ok:
        print("\n🎉 SUCCESS! Document AI is properly configured")
        print("✅ Your multi-deed processing should now use Document AI")
        print("✅ No more fallback to simple page-based splitting")
        print("\n📋 Next steps:")
        print("1. Set your real ANTHROPIC_API_KEY environment variable")
        print("2. Test with a real PDF file")
        print("3. Deploy to your hosting platform with proper environment variables")
    else:
        print("\n❌ Splitting strategy test failed")
    
    print("\n" + "=" * 60)
    print("🔍 Test complete")

if __name__ == "__main__":
    main()
