#!/usr/bin/env python3
"""
Test Integration Script
Test the complete integration of smart chunking into the main app
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mineral_rights.document_classifier import DocumentProcessor
from mineral_rights.document_ai_service import create_document_ai_service

def test_smart_chunking_integration():
    """Test the complete smart chunking integration"""
    
    print("🧪 Testing Smart Chunking Integration")
    print("=" * 50)
    
    # Test with a small PDF first
    test_pdf = "data/multi-deed/pdfs/FRANCO.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    try:
        # Initialize processor
        print("🔧 Initializing DocumentProcessor...")
        processor = DocumentProcessor()
        
        # Test multi-deed processing with smart chunking
        print(f"📄 Processing {test_pdf} with smart chunking...")
        results = processor.process_multi_deed_document(test_pdf, strategy="document_ai")
        
        print(f"✅ Processing completed!")
        print(f"📊 Results:")
        print(f"   - Total deeds processed: {len(results)}")
        
        # Show deed detection details
        if hasattr(processor, '_last_split_result') and processor._last_split_result:
            split_result = processor._last_split_result
            print(f"   - Deeds detected: {split_result.total_deeds}")
            print(f"   - Processing time: {split_result.processing_time:.2f}s")
            
            if split_result.raw_response:
                print(f"   - Chunks processed: {split_result.raw_response.get('chunks_processed', 'N/A')}")
                print(f"   - Systematic offset: {split_result.raw_response.get('systematic_offset', 'N/A')}")
                print(f"   - Raw deeds before merge: {split_result.raw_response.get('raw_deeds_before_merge', 'N/A')}")
            
            print(f"\n📋 Deed Boundaries:")
            for i, deed in enumerate(split_result.deeds):
                start_page = min(deed.pages) + 1  # Convert to 1-indexed
                end_page = max(deed.pages) + 1    # Convert to 1-indexed
                print(f"   Deed {i+1}: Pages {start_page}-{end_page} (Confidence: {deed.confidence:.3f})")
        
        # Show classification results
        print(f"\n🔍 Classification Results:")
        for i, result in enumerate(results):
            if isinstance(result, dict):
                classification = result.get('classification', 'Unknown')
                confidence = result.get('confidence', 0)
                prediction = result.get('prediction', 'Unknown')
                print(f"   Deed {i+1}: {prediction} (Confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_compatibility():
    """Test that the integration works with the API structure"""
    
    print("\n🌐 Testing API Compatibility")
    print("=" * 50)
    
    try:
        # Test the same flow that the API would use
        processor = DocumentProcessor()
        
        # Simulate API call
        test_pdf = "data/multi-deed/pdfs/FRANCO.pdf"
        splitting_strategy = "document_ai"
        
        print(f"📡 Simulating API call with strategy: {splitting_strategy}")
        
        # This is exactly what the API does
        deed_results = processor.process_multi_deed_document(
            test_pdf, 
            strategy=splitting_strategy
        )
        
        # Validate results structure (what API expects)
        if not isinstance(deed_results, list):
            raise Exception(f"Expected list of results, got {type(deed_results)}")
        
        # Wrap results in expected structure (what API does)
        response = {
            "deed_results": deed_results,
            "total_deeds": len(deed_results),
            "summary": {
                "reservations_found": sum(1 for deed in deed_results if deed.get('classification') == 1)
            }
        }
        
        print(f"✅ API compatibility test passed!")
        print(f"📊 Response structure:")
        print(f"   - deed_results: {len(response['deed_results'])} items")
        print(f"   - total_deeds: {response['total_deeds']}")
        print(f"   - reservations_found: {response['summary']['reservations_found']}")
        
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Smart Chunking Integration Test")
    print("=" * 60)
    
    # Test 1: Basic integration
    success1 = test_smart_chunking_integration()
    
    # Test 2: API compatibility
    success2 = test_api_compatibility()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All integration tests passed!")
        print("✅ Smart chunking is ready for production use")
    else:
        print("❌ Some integration tests failed")
        print("🔧 Please check the errors above")
    
    print("=" * 60)
