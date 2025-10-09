#!/usr/bin/env python3
"""
Large PDF Processor - Page-by-page mineral rights detection
==========================================================

Simple, practical approach for processing large PDFs (50-400 pages):
- Process each page individually using existing DocumentProcessor
- Return CSV with pages containing mineral rights reservations
- No complex chunking or memory management
"""

import os
import csv
import time
import fitz
from typing import List, Dict, Any
from .document_classifier import DocumentProcessor

class LargePDFProcessor:
    """Simple page-by-page processor for large PDFs using existing classification logic"""
    
    def __init__(self, api_key: str):
        self.processor = DocumentProcessor(api_key=api_key)
    
    def process_large_pdf(self, pdf_path: str, output_csv: str = None) -> Dict[str, Any]:
        """
        Process large PDF page by page and return results
        
        Args:
            pdf_path: Path to PDF file
            output_csv: Optional CSV file path to save results
            
        Returns:
            Dict with pages containing mineral rights reservations
        """
        print(f"🔍 Processing large PDF: {pdf_path}")
        
        # Get PDF info
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"📄 PDF has {total_pages} pages")
        
        # Process each page using existing page-by-page method
        print(f"🔧 Using existing DocumentProcessor page-by-page method...")
        
        # Use the existing process_document_page_by_page method
        result = self.processor.process_document_page_by_page(
            pdf_path=pdf_path,
            max_samples=3,  # Reduced for speed
            confidence_threshold=0.6,
            max_tokens_per_page=8000,
            high_recall_mode=True
        )
        
        # Extract pages with reservations
        pages_with_reservations = result.get('pages_with_reservations', [])
        page_results = result.get('page_results', [])
        
        # Create simplified results for CSV
        results = []
        for page_result in page_results:
            if page_result.get('has_reservations', False):
                results.append({
                    'page_number': page_result['page_number'],
                    'confidence': page_result['confidence'],
                    'reasoning': page_result.get('reasoning', 'No reasoning provided'),
                    'text_preview': page_result.get('page_text', '')[:200] + "..." if len(page_result.get('page_text', '')) > 200 else page_result.get('page_text', '')
                })
        
        # Create summary
        summary = {
            'total_pages': total_pages,
            'pages_with_reservations': len(pages_with_reservations),
            'reservation_pages': pages_with_reservations,
            'results': results,
            'processing_time': result.get('total_processing_time', 0),
            'classification': result.get('classification', 0),
            'confidence': result.get('confidence', 0.0)
        }
        
        # Save to CSV if requested
        if output_csv:
            self._save_to_csv(results, output_csv)
            print(f"💾 Results saved to: {output_csv}")
        
        print(f"✅ Processing complete: {len(pages_with_reservations)} pages with mineral rights reservations")
        return summary
    
    def _save_to_csv(self, results: List[Dict], csv_path: str):
        """Save results to CSV file"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['page_number', 'confidence', 'reasoning', 'text_preview']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
    
    def process_large_pdf_from_gcs(self, gcs_url: str, output_csv: str = None) -> Dict[str, Any]:
        """
        Process large PDF from GCS URL
        
        Args:
            gcs_url: GCS URL of the PDF
            output_csv: Optional CSV file path to save results
            
        Returns:
            Dict with pages containing mineral rights reservations
        """
        print(f"🔍 Processing large PDF from GCS: {gcs_url}")
        
        # Validate GCS URL format
        if not gcs_url.startswith('https://storage.googleapis.com/'):
            raise ValueError(f"Invalid GCS URL format: {gcs_url}")
        
        # Download from GCS
        from google.cloud import storage
        import tempfile
        
        # Initialize GCS client with proper credentials
        credentials_b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if credentials_b64:
            import base64
            import json
            from google.oauth2 import service_account
            
            # Decode the base64 credentials
            credentials_json = base64.b64decode(credentials_b64).decode('utf-8')
            credentials_info = json.loads(credentials_json)
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            client = storage.Client(credentials=credentials)
            print("✅ Using base64 encoded service account credentials for GCS download")
        else:
            # Fallback to default credentials
            client = storage.Client()
            print("✅ Using default service account credentials for GCS download")
        
        try:
            # Parse GCS URL - handle both formats
            if 'storage.googleapis.com' in gcs_url:
                # Format: https://storage.googleapis.com/bucket-name/path/to/file
                url_parts = gcs_url.split('/')
                if len(url_parts) < 5:
                    raise ValueError(f"Invalid GCS URL format: {gcs_url}")
                
                bucket_name = url_parts[3]
                blob_name = '/'.join(url_parts[4:])
            else:
                # Format: gs://bucket-name/path/to/file
                if gcs_url.startswith('gs://'):
                    gcs_url = gcs_url[5:]  # Remove gs:// prefix
                
                parts = gcs_url.split('/', 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid GCS URL format: {gcs_url}")
                
                bucket_name = parts[0]
                blob_name = parts[1]
            
            print(f"📦 Bucket: {bucket_name}")
            print(f"📄 Blob: {blob_name}")
            
            # Download to temp file
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                blob.download_to_filename(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            try:
                # Process the downloaded file
                result = self.process_large_pdf(tmp_file_path, output_csv)
                return result
            finally:
                # Clean up temp file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            print(f"❌ Error processing GCS file: {e}")
            raise
    
    def process_large_pdf_local(self, pdf_path: str, output_csv: str = None) -> Dict[str, Any]:
        """
        Process large PDF from local file system
        
        Args:
            pdf_path: Local path to PDF file
            output_csv: Optional CSV file path to save results
            
        Returns:
            Dict with pages containing mineral rights reservations
        """
        print(f"🔍 Processing large PDF locally: {pdf_path}")
        
        # Process the local file
        result = self.process_large_pdf(pdf_path, output_csv)
        return result
