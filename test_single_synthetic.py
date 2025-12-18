#!/usr/bin/env python3
"""
Simple test script for a single synthetic file
"""

import os
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append('/Users/lauragomez/Desktop/mineral-rights/src')

from mineral_rights.large_pdf_processor import LargePDFProcessor

def test_single_synthetic_file():
    """Test a single synthetic file"""
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return
    
    # Test file paths
    test_dir = "/Users/lauragomez/Desktop/mineral-rights/data/synthetic_dataset/test"
    pdf_path = os.path.join(test_dir, "pdfs", "synthetic_test_001.pdf")
    ground_truth_path = os.path.join(test_dir, "labels", "synthetic_test_001.json")
    
    print(f"🔍 Testing: {pdf_path}")
    
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"📄 Ground Truth Info:")
    print(f"   - Total pages: {ground_truth['attributes']['total_pages']}")
    print(f"   - Has reservations: {ground_truth['attributes']['has_oil_gas_reservations']}")
    print(f"   - Reservation count: {ground_truth['attributes']['reservation_count']}")
    print(f"   - Total deeds: {ground_truth['attributes']['total_deeds']}")
    
    # Extract expected pages with reservations
    expected_pages = []
    for deed in ground_truth['deeds']:
        if deed['has_oil_gas_reservations']:
            for page in range(deed['page_start'], deed['page_end'] + 1):
                expected_pages.append(page)
    
    expected_pages = sorted(expected_pages)
    print(f"🎯 Expected pages with reservations: {expected_pages}")
    
    try:
        # Initialize processor
        print(f"\n🚀 Initializing LargePDFProcessor...")
        processor = LargePDFProcessor(api_key=api_key)
        
        # Process the PDF locally (not from GCS)
        print(f"📄 Processing PDF locally...")
        result = processor.process_large_pdf_local(pdf_path)
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"   - Total pages processed: {result.get('total_pages', 'N/A')}")
        print(f"   - Pages with reservations: {result.get('pages_with_reservations', 'N/A')}")
        print(f"   - Reservation pages: {result.get('reservation_pages', [])}")
        print(f"   - Processing method: {result.get('processing_method', 'N/A')}")
        
        # Compare with ground truth
        predicted_pages = result.get('reservation_pages', [])
        
        print(f"\n🔍 COMPARISON:")
        print(f"   - Expected pages: {expected_pages}")
        print(f"   - Predicted pages: {predicted_pages}")
        
        # Calculate basic metrics
        true_positives = len(set(expected_pages) & set(predicted_pages))
        false_positives = len(set(predicted_pages) - set(expected_pages))
        false_negatives = len(set(expected_pages) - set(predicted_pages))
        
        precision = true_positives / len(predicted_pages) if predicted_pages else 0
        recall = true_positives / len(expected_pages) if expected_pages else 0
        
        print(f"\n📈 METRICS:")
        print(f"   - True Positives: {true_positives}")
        print(f"   - False Positives: {false_positives}")
        print(f"   - False Negatives: {false_negatives}")
        print(f"   - Precision: {precision:.3f}")
        print(f"   - Recall: {recall:.3f}")
        
        # Show detailed results if available
        if 'results' in result:
            print(f"\n📋 DETAILED RESULTS:")
            for i, res in enumerate(result['results'][:3]):  # Show first 3
                print(f"   Page {res['page_number']}: Confidence {res['confidence']:.3f}")
                print(f"   Reasoning: {res['reasoning'][:100]}...")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_single_synthetic_file()
