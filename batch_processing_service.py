#!/usr/bin/env python3
"""
Batch Processing Service for Document AI

This service implements batch processing to handle large PDFs without chunking.
"""

import os
import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tempfile
import json

@dataclass
class BatchDeedResult:
    """Result from batch processing"""
    deed_number: int
    start_page: int
    end_page: int
    confidence: float
    entity_type: str

class BatchProcessingService:
    """Service for batch processing with Document AI"""
    
    def __init__(self, document_ai_service):
        """
        Initialize batch processing service
        
        Args:
            document_ai_service: The Document AI service to use
        """
        self.document_ai_service = document_ai_service
    
    def process_pdf_batch(self, pdf_path: str) -> List[BatchDeedResult]:
        """
        Process a PDF using batch processing (up to 1000 pages)
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of BatchDeedResult objects
        """
        start_time = time.time()
        
        try:
            print(f"🔄 Batch processing PDF: {pdf_path}")
            
            # Get PDF info
            import fitz
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()
            
            print(f"📄 PDF has {total_pages} pages")
            
            if total_pages > 1000:
                print("❌ PDF exceeds 1000 page limit for batch processing")
                return []
            
            # Process with Document AI directly (no chunking)
            print("🤖 Processing entire PDF with Document AI...")
            split_result = self.document_ai_service.split_deeds_from_pdf(pdf_path, force_single_chunk=True)
            
            # Convert to batch deed results
            batch_deeds = []
            for deed in split_result.deeds:
                batch_deeds.append(BatchDeedResult(
                    deed_number=deed.deed_number,
                    start_page=min(deed.pages) + 1,  # Convert to 1-indexed
                    end_page=max(deed.pages) + 1,    # Convert to 1-indexed
                    confidence=deed.confidence,
                    entity_type='DEED'
                ))
            
            processing_time = time.time() - start_time
            
            print(f"\n📊 BATCH PROCESSING SUMMARY:")
            print(f"✅ Total deeds found: {len(batch_deeds)}")
            print(f"⏱️ Processing time: {processing_time:.2f}s")
            
            return batch_deeds
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            raise
    
    def save_predictions(self, predictions: List[BatchDeedResult], output_file: str = "batch_predictions.json"):
        """Save predictions to JSON file"""
        try:
            # Convert to simple format
            data = []
            for deed in predictions:
                data.append({
                    'deed_number': deed.deed_number,
                    'start_page': deed.start_page,
                    'end_page': deed.end_page,
                    'confidence': deed.confidence,
                    'entity_type': deed.entity_type
                })
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Predictions saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")

def test_batch_processing():
    """Test the batch processing service"""
    print("🧪 Testing batch processing service...")
    
    try:
        # Add the src directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from mineral_rights.document_ai_service import create_document_ai_service
        from batch_processing_service import BatchProcessingService
        
        # Create services
        processor_endpoint = "https://us-documentai.googleapis.com/v1/projects/381937358877/locations/us/processors/895767ed7f252878:process"
        document_ai_service = create_document_ai_service(processor_endpoint)
        batch_service = BatchProcessingService(document_ai_service)
        
        print("✅ Services created successfully")
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        print(f"📄 Testing with PDF: {test_pdf_path}")
        
        # Process with batch processing
        predictions = batch_service.process_pdf_batch(test_pdf_path)
        
        print(f"\n📊 BATCH PROCESSING RESULTS:")
        print(f"✅ Total deeds found: {len(predictions)}")
        
        print(f"\n🔍 Deed details:")
        for deed in predictions:
            print(f"   - Deed {deed.deed_number}: Pages {deed.start_page}-{deed.end_page} (confidence: {deed.confidence:.3f})")
        
        # Save predictions
        batch_service.save_predictions(predictions, "batch_predictions.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_processing()
    if success:
        print("\n🎉 Batch processing service is working!")
        print("📊 Summary:")
        print("   - PDF processed without chunking")
        print("   - Entire document processed as one unit")
        print("   - Predictions saved for comparison")
    else:
        print("\n❌ Test failed")
    
    exit(0 if success else 1)
