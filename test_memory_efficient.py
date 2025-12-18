#!/usr/bin/env python3
"""
Test the memory-efficient streaming processor
"""
import os
import sys
sys.path.append('src')

from mineral_rights.memory_efficient_processor import MemoryEfficientProcessor

def test_memory_efficient():
    """Test the memory-efficient processor with a small PDF"""
    
    # Set API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        return False
    
    # Test with synthetic PDF
    pdf_path = "data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf"
    output_csv = "test_memory_efficient_results.csv"
    
    print("🧪 Testing Memory-Efficient Streaming Processor")
    print(f"📄 Input PDF: {pdf_path}")
    print(f"📊 Output CSV: {output_csv}")
    
    try:
        # Initialize processor
        processor = MemoryEfficientProcessor(api_key=api_key)
        
        # Process with streaming
        result = processor.process_pdf_streaming(pdf_path, output_csv)
        
        print("\n✅ RESULTS:")
        print(f"📊 Total pages: {result['total_pages']}")
        print(f"🎯 Pages with reservations: {result['pages_with_reservations']}")
        print(f"📄 Reservation pages: {result['reservation_pages']}")
        print(f"💾 Processing method: {result['processing_method']}")
        
        if os.path.exists(output_csv):
            print(f"📁 CSV saved to: {output_csv}")
            # Show first few lines
            with open(output_csv, 'r') as f:
                lines = f.readlines()
                print(f"📋 CSV has {len(lines)-1} data rows")
                if len(lines) > 1:
                    print("📋 First row:", lines[1].strip())
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_efficient()
    if success:
        print("\n🎉 Memory-efficient test completed successfully!")
    else:
        print("\n💥 Memory-efficient test failed!")
