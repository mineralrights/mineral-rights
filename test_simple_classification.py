#!/usr/bin/env python3
"""
Simple test to verify classification is working
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mineral_rights.document_classifier import DocumentProcessor

def test_simple_classification():
    """Test with a simple single-page PDF first"""
    
    print("🧪 SIMPLE CLASSIFICATION TEST")
    print("=" * 40)
    
    # Check environment variables
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False
    
    print("✅ Environment variables found")
    
    # Initialize processor
    try:
        print("\n🔧 Initializing DocumentProcessor...")
        processor = DocumentProcessor(api_key=api_key)
        print("✅ Processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Test with a simple PDF first
    test_pdf = "test_small.pdf"
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    print(f"\n📄 Testing with: {test_pdf}")
    
    try:
        print("\n🚀 Starting simple classification...")
        result = processor.process_document(test_pdf, max_samples=3, high_recall_mode=True)
        
        print(f"\n📊 RESULT:")
        print(f"   - Classification: {result.get('classification', 'Unknown')}")
        print(f"   - Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"   - Samples used: {result.get('samples_used', 0)}")
        print(f"   - Early stopped: {result.get('early_stopped', False)}")
        
        # Get reasoning
        detailed_samples = result.get('detailed_samples', [])
        if detailed_samples:
            reasoning = detailed_samples[0].get('reasoning', 'No reasoning')
            print(f"   - Reasoning: {reasoning[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_classification()
    if success:
        print(f"\n✅ SIMPLE TEST PASSED - Basic classification works!")
    else:
        print(f"\n❌ SIMPLE TEST FAILED")
        sys.exit(1)
