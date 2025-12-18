#!/usr/bin/env python3
"""
Test script for evaluating page-by-page processing on synthetic dataset
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

# Add the src directory to the path
sys.path.append('/Users/lauragomez/Desktop/mineral-rights/src')

from mineral_rights.large_pdf_processor import LargePDFProcessor

def load_ground_truth(ground_truth_path: str) -> Dict[str, Any]:
    """Load ground truth from JSON file"""
    with open(ground_truth_path, 'r') as f:
        return json.load(f)

def extract_ground_truth_pages(ground_truth: Dict[str, Any]) -> List[int]:
    """Extract pages with mineral rights reservations from ground truth"""
    pages_with_reservations = []
    
    for deed in ground_truth['deeds']:
        if deed['has_oil_gas_reservations']:
            # Add all pages in this deed range
            for page in range(deed['page_start'], deed['page_end'] + 1):
                pages_with_reservations.append(page)
    
    return sorted(pages_with_reservations)

def test_single_file(pdf_path: str, ground_truth_path: str, api_key: str) -> Dict[str, Any]:
    """Test a single PDF file and compare with ground truth"""
    print(f"\n🔍 Testing: {os.path.basename(pdf_path)}")
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)
    expected_pages = extract_ground_truth_pages(ground_truth)
    
    print(f"📄 Total pages: {ground_truth['attributes']['total_pages']}")
    print(f"🎯 Expected pages with reservations: {expected_pages}")
    print(f"📊 Expected reservation count: {ground_truth['attributes']['reservation_count']}")
    
    try:
        # Initialize processor
        processor = LargePDFProcessor(api_key=api_key)
        
        # Process the PDF
        result = processor.process_large_pdf_from_gcs(pdf_path)
        
        # Extract predicted pages
        predicted_pages = result.get('reservation_pages', [])
        
        print(f"🤖 Predicted pages with reservations: {predicted_pages}")
        print(f"📊 Predicted reservation count: {len(predicted_pages)}")
        
        # Calculate metrics
        true_positives = len(set(expected_pages) & set(predicted_pages))
        false_positives = len(set(predicted_pages) - set(expected_pages))
        false_negatives = len(set(expected_pages) - set(predicted_pages))
        
        precision = true_positives / len(predicted_pages) if predicted_pages else 0
        recall = true_positives / len(expected_pages) if expected_pages else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Page-level accuracy
        total_pages = ground_truth['attributes']['total_pages']
        correct_predictions = total_pages - false_positives - false_negatives
        page_accuracy = correct_predictions / total_pages
        
        metrics = {
            'file': os.path.basename(pdf_path),
            'total_pages': total_pages,
            'expected_pages': expected_pages,
            'predicted_pages': predicted_pages,
            'expected_count': len(expected_pages),
            'predicted_count': len(predicted_pages),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'page_accuracy': page_accuracy,
            'success': True
        }
        
        print(f"✅ Precision: {precision:.3f}")
        print(f"✅ Recall: {recall:.3f}")
        print(f"✅ F1 Score: {f1_score:.3f}")
        print(f"✅ Page Accuracy: {page_accuracy:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return {
            'file': os.path.basename(pdf_path),
            'error': str(e),
            'success': False
        }

def test_synthetic_dataset(test_dir: str, api_key: str, max_files: int = 5) -> Dict[str, Any]:
    """Test the synthetic dataset"""
    print(f"🚀 Testing synthetic dataset with max {max_files} files")
    
    test_path = Path(test_dir)
    pdfs_dir = test_path / "pdfs"
    labels_dir = test_path / "labels"
    
    results = []
    successful_tests = 0
    
    # Get list of PDF files
    pdf_files = sorted(list(pdfs_dir.glob("*.pdf")))[:max_files]
    
    for pdf_file in pdf_files:
        # Find corresponding ground truth file
        label_file = labels_dir / f"{pdf_file.stem}.json"
        
        if not label_file.exists():
            print(f"⚠️ No ground truth found for {pdf_file.name}")
            continue
        
        # Test the file
        result = test_single_file(str(pdf_file), str(label_file), api_key)
        results.append(result)
        
        if result.get('success', False):
            successful_tests += 1
    
    # Calculate overall metrics
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
        avg_f1 = sum(r['f1_score'] for r in successful_results) / len(successful_results)
        avg_accuracy = sum(r['page_accuracy'] for r in successful_results) / len(successful_results)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"✅ Successful tests: {successful_tests}/{len(results)}")
        print(f"✅ Average Precision: {avg_precision:.3f}")
        print(f"✅ Average Recall: {avg_recall:.3f}")
        print(f"✅ Average F1 Score: {avg_f1:.3f}")
        print(f"✅ Average Page Accuracy: {avg_accuracy:.3f}")
        
        # Save detailed results
        results_df = pd.DataFrame(successful_results)
        results_df.to_csv('synthetic_dataset_results.csv', index=False)
        print(f"💾 Detailed results saved to: synthetic_dataset_results.csv")
        
        return {
            'total_tests': len(results),
            'successful_tests': successful_tests,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'avg_page_accuracy': avg_accuracy,
            'detailed_results': successful_results
        }
    else:
        print("❌ No successful tests completed")
        return {'error': 'No successful tests'}

def main():
    """Main test function"""
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return
    
    # Test directory
    test_dir = "/Users/lauragomez/Desktop/mineral-rights/data/synthetic_dataset/test"
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"🔍 Testing synthetic dataset in: {test_dir}")
    
    # Run tests
    results = test_synthetic_dataset(test_dir, api_key, max_files=5)
    
    if 'error' not in results:
        print(f"\n🎉 Testing completed successfully!")
        print(f"📈 Overall Performance:")
        print(f"   - Precision: {results['avg_precision']:.3f}")
        print(f"   - Recall: {results['avg_recall']:.3f}")
        print(f"   - F1 Score: {results['avg_f1_score']:.3f}")
        print(f"   - Page Accuracy: {results['avg_page_accuracy']:.3f}")

if __name__ == "__main__":
    main()
