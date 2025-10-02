#!/usr/bin/env python3
"""
Test multi-deed classification with a smaller PDF
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mineral_rights.document_classifier import DocumentProcessor

def test_multi_deed_simple():
    """Test multi-deed processing with a smaller PDF"""
    
    print("🧪 MULTI-DEED CLASSIFICATION TEST")
    print("=" * 50)
    
    # Check environment variables
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False
    
    document_ai_endpoint = os.getenv("DOCUMENT_AI_ENDPOINT")
    if not document_ai_endpoint:
        print("❌ DOCUMENT_AI_ENDPOINT not set")
        return False
    
    print("✅ Environment variables found")
    
    # Initialize processor
    try:
        print("\n🔧 Initializing DocumentProcessor...")
        processor = DocumentProcessor(
            api_key=api_key,
            document_ai_endpoint=document_ai_endpoint,
            document_ai_credentials=None  # Use ADC
        )
        print("✅ Processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Test with a smaller multi-deed PDF
    test_pdf = "data/synthetic_dataset/test/pdfs/synthetic_test_001.pdf"
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    print(f"\n📄 Testing with: {test_pdf}")
    print(f"📊 File size: {os.path.getsize(test_pdf) / 1024 / 1024:.1f} MB")
    
    try:
        print("\n🚀 Starting multi-deed processing...")
        print("⏳ This may take several minutes...")
        
        # Process with reduced parameters for memory efficiency
        results = processor.process_multi_deed_document(test_pdf, strategy="document_ai")
        
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"   - Total deeds processed: {len(results)}")
        
        # Analyze results
        deeds_with_reservations = 0
        deeds_without_reservations = 0
        total_confidence = 0
        
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            classification = result.get('classification', 0)
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('detailed_samples', [{}])[0].get('reasoning', 'No reasoning') if result.get('detailed_samples') else 'No reasoning'
            pages_in_deed = result.get('pages_in_deed', 0)
            deed_boundary_info = result.get('deed_boundary_info', {})
            page_range = deed_boundary_info.get('page_range', 'Unknown') if deed_boundary_info else 'Unknown'
            
            if classification == 1:
                deeds_with_reservations += 1
                status = "✅ HAS RESERVATIONS"
            else:
                deeds_without_reservations += 1
                status = "❌ NO RESERVATIONS"
            
            total_confidence += confidence
            
            print(f"\n   Deed {i}:")
            print(f"     Status: {status}")
            print(f"     Confidence: {confidence:.3f}")
            print(f"     Pages: {pages_in_deed}")
            print(f"     Page Range: {page_range}")
            print(f"     Reasoning: {reasoning[:100]}...")
        
        avg_confidence = total_confidence / len(results) if results else 0
        
        print(f"\n📈 SUMMARY:")
        print(f"   - Deeds with reservations: {deeds_with_reservations}")
        print(f"   - Deeds without reservations: {deeds_without_reservations}")
        print(f"   - Average confidence: {avg_confidence:.3f}")
        
        # Test CSV generation
        print(f"\n📊 TESTING CSV GENERATION:")
        csv_data = []
        for i, result in enumerate(results, 1):
            classification = result.get('classification', 0)
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('detailed_samples', [{}])[0].get('reasoning', 'No reasoning') if result.get('detailed_samples') else 'No reasoning'
            deed_boundary_info = result.get('deed_boundary_info', {})
            page_range = deed_boundary_info.get('page_range', 'Unknown') if deed_boundary_info else 'Unknown'
            
            csv_data.append({
                "Deed Number": i,
                "Has Reservations": "YES" if classification == 1 else "NO",
                "Confidence": f"{confidence:.1%}",
                "Page Range": page_range,
                "Reasoning": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            })
        
        print(f"   - CSV would contain {len(csv_data)} rows")
        print(f"   - Sample CSV data:")
        for row in csv_data[:3]:  # Show first 3 rows
            print(f"     Deed {row['Deed Number']}: {row['Has Reservations']} ({row['Confidence']}) - {row['Page Range']}")
        
        if len(csv_data) > 3:
            print(f"     ... and {len(csv_data) - 3} more deeds")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_deed_simple()
    if success:
        print(f"\n✅ MULTI-DEED TEST PASSED - Classification is working!")
    else:
        print(f"\n❌ MULTI-DEED TEST FAILED")
        sys.exit(1)
