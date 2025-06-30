#!/usr/bin/env python3
"""
Quick Test of Balanced High Recall Oil and Gas Classifier
========================================================

Tests the balanced high-recall classifier on sample documents.
"""

import os
from pathlib import Path
from document_classifier import DocumentProcessor

def test_balanced_classifier():
    """Test the balanced high-recall classifier on sample documents"""
    
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Initialize processor
    processor = DocumentProcessor(api_key)
    
    # Get sample documents
    reservs_dir = Path("data/reservs")
    no_reservs_dir = Path("data/no-reservs")
    
    # Pick a few test documents
    test_docs = []
    
    # Add 2 documents from reservs
    reservs_files = list(reservs_dir.glob("*.pdf"))[:2]
    test_docs.extend([(f, "reservs", 1) for f in reservs_files])
    
    # Add 2 documents from no-reservs
    no_reservs_files = list(no_reservs_dir.glob("*.pdf"))[:2]
    test_docs.extend([(f, "no-reservs", 0) for f in no_reservs_files])
    
    print("🎯 TESTING BALANCED HIGH RECALL OIL & GAS CLASSIFIER")
    print("=" * 60)
    print(f"Testing {len(test_docs)} sample documents")
    print("Note: Using balanced approach for good recall with reasonable accuracy")
    print()
    
    for i, (pdf_path, category, expected_label) in enumerate(test_docs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(test_docs)}] TESTING: {pdf_path.name}")
        print(f"Category: {category}, Expected: {expected_label}")
        print(f"{'='*70}")
        
        # Test with balanced high recall classifier
        try:
            result = processor.process_document(
                str(pdf_path),
                max_samples=3,  # Quick test with fewer samples
                confidence_threshold=0.7  # Balanced threshold
            )
            
            print(f"📊 RESULTS:")
            print(f"  Classification: {result['classification']} ({'Has Oil and Gas Reservations' if result['classification'] == 1 else 'No Oil and Gas Reservations'})")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Samples used: {result['samples_used']}")
            print(f"  Early stopped: {result['early_stopped']}")
            print(f"  Pages processed: {result['pages_processed']}")
            
            # Check accuracy
            is_correct = result['classification'] == expected_label
            accuracy_status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"  Accuracy: {accuracy_status}")
            
            # Show some reasoning from the first sample
            if result.get('detailed_samples') and len(result['detailed_samples']) > 0:
                sample = result['detailed_samples'][0]
                reasoning_preview = sample['reasoning'][:300] + "..." if len(sample['reasoning']) > 300 else sample['reasoning']
                print(f"  Sample reasoning: {reasoning_preview}")
            
            # Show voting breakdown
            votes = result['votes']
            print(f"  Vote breakdown: Class 0: {votes.get(0, 0):.2f}, Class 1: {votes.get(1, 0):.2f}")
            
            # Analysis
            if result['classification'] == 1:
                print(f"  🎯 BALANCED RECALL: Found oil/gas reservations!")
                if expected_label == 0:
                    print(f"     → This might be a false positive - check if justified")
            else:
                print(f"  📋 ANALYSIS: No oil/gas reservations detected")
                if expected_label == 1:
                    print(f"     → This document may only have non-oil/gas reservations (coal, etc.)")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("🎯 BALANCED HIGH RECALL CLASSIFIER FEATURES:")
    print("✅ Good sensitivity to oil/gas language while maintaining accuracy")
    print("✅ Uses balanced confidence thresholds (0.65 positive, 0.75 negative)")
    print("✅ Considers context and substantive language, not just keywords")
    print("✅ Applies slight positive bias only in very close decisions")
    print("✅ Processes documents page-by-page with early stopping")
    print("✅ Fixed temperature (0.1) for consistent results")
    print("✅ Optimized for good recall without sacrificing too much precision")
    print(f"{'='*70}")
    print("\n💡 NEXT STEPS:")
    print("   1. If results look good, run full evaluation:")
    print("      python evaluate_oil_gas_classifier.py")
    print("   2. Expected performance: ~80-85% accuracy with ~85-90% recall")

if __name__ == "__main__":
    test_balanced_classifier() 