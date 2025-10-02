#!/usr/bin/env python3
"""
Local testing script for mineral rights processor
Tests the complete pipeline: PDF splitting + classification
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mineral_rights.document_classifier import DocumentProcessor

def test_complete_pipeline(pdf_path: str):
    """Test the complete pipeline with a PDF"""
    
    print(f"🧪 Testing complete pipeline with: {pdf_path}")
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
        # Initialize processor
        print("🔧 Initializing document processor...")
        processor = DocumentProcessor(api_key=api_key)
        print("✅ Processor initialized successfully")
        
        # Test 1: Single deed processing (if it's a single deed)
        print("\n📄 Test 1: Single deed processing...")
        start_time = time.time()
        
        result = processor.process_document_memory_efficient(
            pdf_path,
            chunk_size=25,  # Small chunks for testing
            max_samples=4,  # Fewer samples for speed
            high_recall_mode=True
        )
        
        single_time = time.time() - start_time
        print(f"✅ Single deed processing completed in {single_time:.1f}s")
        print(f"   Classification: {result['classification']} ({'Has reservations' if result['classification'] == 1 else 'No reservations'})")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Pages processed: {result['pages_processed']}")
        
        # Test 2: Multi-deed processing (if it's a multi-deed document)
        print("\n📚 Test 2: Multi-deed processing...")
        start_time = time.time()
        
        multi_result = processor.process_multi_deed_document(
            pdf_path,
            strategy="document_ai"  # Use Document AI if available, fallback to simple
        )
        
        multi_time = time.time() - start_time
        print(f"✅ Multi-deed processing completed in {multi_time:.1f}s")
        print(f"   Total deeds processed: {len(multi_result)}")
        
        # Show results for each deed
        for i, deed_result in enumerate(multi_result):
            print(f"   Deed {i+1}: {deed_result['classification']} (confidence: {deed_result['confidence']:.3f})")
            if 'error' in deed_result:
                print(f"     Error: {deed_result['error']}")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Single deed time: {single_time:.1f}s")
        print(f"   Multi-deed time: {multi_time:.1f}s")
        print(f"   Total deeds found: {len(multi_result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main testing function"""
    
    print("🚀 Mineral Rights Processor - Local Testing")
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
            print("   python test_local.py /path/to/your/pdf")
            return
    
    print(f"📁 Using PDF: {pdf_path}")
    
    # Check file size
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    print(f"📊 File size: {file_size:.1f} MB")
    
    if file_size > 100:
        print("⚠️  Large file detected. This may take a while...")
    
    # Run tests
    success = test_complete_pipeline(pdf_path)
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🚀 Ready for Cloud Run deployment!")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
