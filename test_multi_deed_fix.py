#!/usr/bin/env python3
"""
Test script to verify multi-deed processing fixes
"""
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_multi_deed_processing():
    """Test multi-deed processing with memory-efficient methods"""
    print("🔧 Testing Multi-Deed Processing Fixes")
    print("=" * 50)
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        # Initialize processor
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
            
        processor = DocumentProcessor(api_key)
        print("✅ Document processor initialized")
        
        # Test with a small PDF
        test_pdf = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/chunks/BEUCLER/BEUCLER_chunk01_p001-059.pdf"
        if not os.path.exists(test_pdf):
            print(f"❌ Test PDF not found: {test_pdf}")
            return False
            
        print(f"📄 Testing with: {test_pdf}")
        
        # Test multi-deed processing
        print("\n🔧 Testing multi-deed processing...")
        start_time = time.time()
        
        try:
            results = processor.process_multi_deed_document(
                test_pdf, 
                strategy="smart_detection"
            )
            
            processing_time = time.time() - start_time
            print(f"✅ Multi-deed processing completed in {processing_time:.1f}s")
            print(f"📊 Results: {len(results)} deeds processed")
            
            for i, result in enumerate(results):
                print(f"  Deed {i+1}: classification={result.get('classification')}, confidence={result.get('confidence', 0):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Multi-deed processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

def test_memory_efficient_processing():
    """Test memory-efficient processing"""
    print("\n🔧 Testing Memory-Efficient Processing")
    print("=" * 50)
    
    try:
        from mineral_rights.document_classifier import DocumentProcessor
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
            
        processor = DocumentProcessor(api_key)
        
        test_pdf = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/chunks/BEUCLER/BEUCLER_chunk01_p001-059.pdf"
        if not os.path.exists(test_pdf):
            print(f"❌ Test PDF not found: {test_pdf}")
            return False
            
        print(f"📄 Testing memory-efficient processing with: {test_pdf}")
        
        start_time = time.time()
        
        try:
            result = processor.process_document_memory_efficient(
                test_pdf,
                chunk_size=10,  # Small chunks for testing
                max_samples=3,  # Few samples for speed
                high_recall_mode=True
            )
            
            processing_time = time.time() - start_time
            print(f"✅ Memory-efficient processing completed in {processing_time:.1f}s")
            print(f"📊 Result: classification={result.get('classification')}, confidence={result.get('confidence', 0):.3f}")
            print(f"📄 Pages processed: {result.get('pages_processed', 0)}")
            print(f"🧹 Chunks processed: {result.get('chunks_processed', 0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Memory-efficient processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Multi-Deed and Memory-Efficient Processing Fixes")
    print("=" * 60)
    
    success = True
    
    # Test multi-deed processing
    if not test_multi_deed_processing():
        success = False
    
    # Test memory-efficient processing
    if not test_memory_efficient_processing():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! Multi-deed processing should work now.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("\n📋 Summary of fixes implemented:")
    print("1. ✅ Fixed undefined variable bug in memory-efficient processing")
    print("2. ✅ Added error handling to multi-deed processing")
    print("3. ✅ Updated multi-deed to use memory-efficient processing")
    print("4. ✅ Added proper cleanup and error recovery")
    print("5. ✅ Added progress logging for debugging")
