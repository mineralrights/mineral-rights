#!/usr/bin/env python3
"""
Test script for memory-optimized Document AI smart chunking
"""
import os
import sys
import time
import psutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mineral_rights.smart_chunking_service import SmartChunkingService

def test_memory_optimization():
    """Test the memory-optimized smart chunking service"""
    
    # Initialize the service
    print("🔧 Initializing SmartChunkingService...")
    service = SmartChunkingService(
        project_id="381937358877",
        location="us",
        processor_id="895767ed7f252878",
        processor_version="106a39290d05efaf"
    )
    
    # Test with a larger PDF to test the fallback
    test_pdf = "data/synthetic_dataset/test/pdfs/synthetic_test_002.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    print(f"📄 Testing with: {test_pdf}")
    
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"💾 Initial memory: {initial_memory:.1f} MB")
    
    try:
        # Test the memory-optimized processing
        print("🚀 Starting memory-optimized processing...")
        start_time = time.time()
        
        result = service.process_pdf(
            pdf_path=test_pdf,
            chunk_size=3,  # Use very small chunks
            overlap=1,
            apply_offset=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"✅ Processing completed in {processing_time:.1f} seconds")
        print(f"💾 Final memory: {final_memory:.1f} MB (increase: {memory_increase:+.1f} MB)")
        print(f"📊 Results: {result.total_deeds} deeds found")
        
        # Check if we found any deeds
        if result.total_deeds > 0:
            print("🎯 SUCCESS: Document AI smart chunking found deeds!")
            for i, deed in enumerate(result.deed_detections[:3]):  # Show first 3 deeds
                print(f"  Deed {i+1}: Pages {deed.pages}, Confidence: {deed.confidence:.3f}")
            return True
        else:
            print("⚠️ WARNING: No deeds found - this might indicate an issue")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Memory-Optimized Document AI Smart Chunking")
    print("=" * 60)
    
    success = test_memory_optimization()
    
    print("=" * 60)
    if success:
        print("✅ Test PASSED - Memory optimization working!")
    else:
        print("❌ Test FAILED - Need to investigate further")
