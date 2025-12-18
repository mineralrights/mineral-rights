#!/usr/bin/env python3
"""
Quick Test for Oil and Gas Classification
========================================

Tests the updated classifier on a small subset of documents
to verify it's working correctly before running full evaluation.
"""

import os
from pathlib import Path
from mineral_rights.document_classifier import DocumentProcessor

def quick_test():
    """Test on a few documents to verify functionality"""
    
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    try:
        # Initialize processor
        processor = DocumentProcessor(api_key)
        
        # Get a few test files
        reservs_dir = Path("data/reservs")
        no_reservs_dir = Path("data/no-reservs")
        
        reservs_files = list(reservs_dir.glob("*.pdf"))[:2]  # First 2 reservation docs
        no_reservs_files = list(no_reservs_dir.glob("*.pdf"))[:2]  # First 2 no-reservation docs
        
        print("🧪 QUICK OIL AND GAS CLASSIFICATION TEST")
        print("=" * 50)
        print(f"Testing on {len(reservs_files)} reservation docs + {len(no_reservs_files)} no-reservation docs")
        print()
        
        # Test reservation documents
        print("🔥 Testing documents WITH reservations:")
        for i, pdf_path in enumerate(reservs_files, 1):
            print(f"\n[{i}] {pdf_path.name}")
            
            try:
                result = processor.process_document(
                    str(pdf_path),
                    max_samples=2,  # Quick test with fewer samples
                    confidence_threshold=0.7
                )
                
                classification = result['classification']
                confidence = result['confidence']
                
                if classification == 1:
                    status = "✅ FOUND OIL/GAS RESERVATION"
                else:
                    status = "❌ RECLASSIFIED (likely coal/other minerals only)"
                
                print(f"   Result: {classification} - {status}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Samples used: {result['samples_used']}")
                
                # Show a snippet of reasoning from first sample
                if result.get('detailed_samples'):
                    reasoning = result['detailed_samples'][0]['reasoning'][:200]
                    print(f"   Reasoning: {reasoning}...")
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
        
        # Test no-reservation documents
        print(f"\n🚫 Testing documents WITHOUT reservations:")
        for i, pdf_path in enumerate(no_reservs_files, 1):
            print(f"\n[{i}] {pdf_path.name}")
            
            try:
                result = processor.process_document(
                    str(pdf_path),
                    max_samples=2,  # Quick test with fewer samples
                    confidence_threshold=0.7
                )
                
                classification = result['classification']
                confidence = result['confidence']
                
                if classification == 0:
                    status = "✅ CORRECTLY NO OIL/GAS"
                else:
                    status = "❌ FALSE POSITIVE (found oil/gas when none expected)"
                
                print(f"   Result: {classification} - {status}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Samples used: {result['samples_used']}")
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
        
        print(f"\n{'='*50}")
        print("🎯 Quick test completed!")
        print("\nIf results look good, run the full evaluation with:")
        print("   python evaluate_oil_gas_classifier.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    quick_test() 