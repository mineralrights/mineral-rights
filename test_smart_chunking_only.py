#!/usr/bin/env python3
"""
Test Smart Chunking Service Only
Test just the smart chunking service without the full classifier
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mineral_rights.smart_chunking_service import SmartChunkingService

def test_smart_chunking_service():
    """Test the smart chunking service directly"""
    
    print("🧪 Testing Smart Chunking Service")
    print("=" * 50)
    
    # Test with a small PDF first
    test_pdf = "data/multi-deed/pdfs/FRANCO.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    try:
        # Initialize smart chunking service
        print("🔧 Initializing SmartChunkingService...")
        service = SmartChunkingService(
            project_id="381937358877",
            location="us",
            processor_id="895767ed7f252878",
            processor_version="106a39290d05efaf"
        )
        
        # Test processing
        print(f"📄 Processing {test_pdf} with smart chunking...")
        result = service.process_pdf(test_pdf)
        
        print(f"✅ Processing completed!")
        print(f"📊 Results:")
        print(f"   - Total deeds detected: {result.total_deeds}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        print(f"   - Chunks processed: {result.chunks_processed}")
        print(f"   - Systematic offset: {result.systematic_offset}")
        print(f"   - Raw deeds before merge: {result.raw_deeds_before_merge}")
        
        print(f"\n📋 Deed Boundaries:")
        for detection in result.deed_detections:
            print(f"   Deed {detection.deed_number}: Pages {detection.start_page}-{detection.end_page} (Confidence: {detection.confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Smart Chunking Service Test")
    print("=" * 60)
    
    success = test_smart_chunking_service()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Smart chunking service test passed!")
        print("✅ Service is ready for integration")
    else:
        print("❌ Smart chunking service test failed")
        print("🔧 Please check the errors above")
    
    print("=" * 60)
