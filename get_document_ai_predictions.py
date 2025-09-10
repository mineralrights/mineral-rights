#!/usr/bin/env python3
"""
Get Document AI Predictions

This script gets the Document AI predictions so you can compare them to your ground truths.
"""

import os
import sys
from pathlib import Path
import json

def get_document_ai_predictions():
    """Get Document AI predictions for the PDF"""
    print("🤖 Getting Document AI predictions...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from mineral_rights.document_ai_service import create_document_ai_service
        from simple_chunking_solution import SimpleChunkingService
        
        # Create services
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        document_ai_service = create_document_ai_service(processor_endpoint)
        simple_chunking_service = SimpleChunkingService(document_ai_service)
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return None
        
        print(f"📄 Processing PDF: {test_pdf_path}")
        
        # Process with simple chunking
        deeds = simple_chunking_service.process_pdf_simple(test_pdf_path)
        
        # Convert to simple format for comparison
        predictions = []
        for deed in deeds:
            predictions.append({
                'deed_number': deed.deed_number,
                'start_page': deed.start_page + 1,  # Convert to 1-indexed
                'end_page': deed.end_page + 1,      # Convert to 1-indexed
                'confidence': deed.confidence,
                'chunk_id': deed.chunk_id
            })
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        return None

def main():
    """Main function"""
    print("📊 Getting Document AI Predictions")
    print("=" * 40)
    
    # Get predictions
    predictions = get_document_ai_predictions()
    
    if predictions:
        print(f"\n📋 Document AI Predictions:")
        print(f"Total deeds detected: {len(predictions)}")
        print(f"\nDeed details:")
        
        for deed in predictions:
            print(f"   - Deed {deed['deed_number']}: Pages {deed['start_page']}-{deed['end_page']} (confidence: {deed['confidence']:.3f})")
        
        # Save to JSON file for easy comparison
        output_file = "document_ai_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"\n💾 Predictions saved to: {output_file}")
        print(f"📊 You can now compare these with your ground truths")
        
        return True
    else:
        print("❌ Failed to get predictions")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
