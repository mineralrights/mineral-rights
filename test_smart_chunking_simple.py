#!/usr/bin/env python3
"""
Simple test for smart chunking - one PDF at a time to avoid memory issues
"""

import os
import sys
from smart_chunking_processor import SmartChunkingProcessor

def test_single_pdf(pdf_name: str):
    """Test smart chunking on a single PDF"""
    print(f"🧪 Testing Smart Chunking: {pdf_name}")
    print("=" * 50)
    
    # Initialize processor
    processor = SmartChunkingProcessor(
        project_id="381937358877",
        location="us",
        processor_id="895767ed7f252878",
        processor_version="106a39290d05efaf"
    )
    
    pdf_path = f"data/multi-deed/pdfs/{pdf_name}"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return None
    
    try:
        result = processor.process_pdf(pdf_path, max_chunk_size=25, overlap=5)
        
        print(f"\n✅ {pdf_name} completed successfully!")
        print(f"📊 Results:")
        print(f"   - Chunks: {result.total_chunks}")
        print(f"   - Time: {result.total_processing_time:.1f}s")
        print(f"   - Over-detection: {result.over_detection_ratio:.2f}x")
        print(f"   - Systematic offset: {result.systematic_offset}")
        if result.systematic_offset:
            print(f"   - Offset: {result.mean_offset:.1f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {pdf_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_name = sys.argv[1]
    else:
        pdf_name = "FRANCO.pdf"  # Default to FRANCO.pdf
    
    test_single_pdf(pdf_name)
