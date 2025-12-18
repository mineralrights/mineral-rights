#!/usr/bin/env python3
"""
Diagnostic script to check if LLM is being called during processing
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mineral_rights.document_classifier import DocumentProcessor

def test_llm_calls(pdf_path: str):
    """Test if LLM is actually being called"""
    
    print(f"🔍 Testing LLM calls with: {pdf_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize processor
        print("\n🔧 Initializing document processor...")
        processor = DocumentProcessor(api_key=api_key)
        print("✅ Processor initialized successfully")
        
        # Test with a simple process_document call
        print("\n📄 Processing document in single_deed mode...")
        print("   This should call the LLM multiple times...")
        print("   Watch for 'Generating sample X/Y...' messages\n")
        
        result = processor.process_document(
            pdf_path,
            max_samples=3,  # Just 3 samples for quick test
            confidence_threshold=0.7,
            page_strategy="first_few",
            high_recall_mode=True
        )
        
        print("\n" + "=" * 60)
        print("📊 RESULTS:")
        print("=" * 60)
        print(f"   Classification: {result.get('classification', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0.0)}")
        print(f"   Samples used: {result.get('samples_used', 0)}")
        print(f"   Detailed samples: {len(result.get('detailed_samples', []))}")
        
        # Check if samples were generated
        detailed_samples = result.get('detailed_samples', [])
        if len(detailed_samples) == 0:
            print("\n❌ PROBLEM DETECTED: No detailed samples generated!")
            print("   This means the LLM was NOT called successfully.")
            print("   Possible causes:")
            print("   - API key is invalid/expired")
            print("   - Network issues")
            print("   - API rate limits")
            return False
        else:
            print(f"\n✅ LLM was called! Generated {len(detailed_samples)} samples")
            print(f"   First sample reasoning: {detailed_samples[0].get('reasoning', 'N/A')[:200]}...")
            return True
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Use DB 107_12.pdf which should have reservations
    pdf_path = "data/reservs/DB 107_12.pdf"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        print("\nAvailable test PDFs:")
        print("  - data/reservs/DB 107_12.pdf")
        print("  - data/reservs/DB 257_85.pdf")
        print("  - data/no-reservs/DB 110_36.pdf")
        sys.exit(1)
    
    success = test_llm_calls(pdf_path)
    sys.exit(0 if success else 1)

