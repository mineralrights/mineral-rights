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
        self.job_id = None
        self.job_results = None
    
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
            max_samples=1,  # Single sample for maximum speed
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
    
    def process_large_pdf_with_progress(self, pdf_path: str, job_id: str, job_results: dict) -> Dict[str, Any]:
        """
        Process large PDF with real-time progress updates
        
        Args:
            pdf_path: Path to PDF file
            job_id: Job ID for progress tracking
            job_results: Global job results dictionary
            
        Returns:
            Dict with pages containing mineral rights reservations
        """
        self.job_id = job_id
        self.job_results = job_results
        
        print(f"🔍 Processing large PDF with progress tracking: {pdf_path}")
        
        # Get PDF info
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"📄 PDF has {total_pages} pages")
        
        # Update initial progress
        self._update_progress(0, total_pages, [], 0, 0)
        
        # Process each page individually with progress updates
        pages_with_reservations = []
        page_results = []
        start_time = time.time()
        
        for page_num in range(total_pages):
            current_page = page_num + 1
            print(f"\n--- PROCESSING PAGE {current_page}/{total_pages} ---")
            
            # Process single page
            page_result = self._process_single_page_with_progress(pdf_path, page_num, current_page)
            page_results.append(page_result)
            
            # Track results
            if page_result.get('has_reservations', False):
                pages_with_reservations.append(current_page)
                print(f"🎯 PAGE {current_page}: HAS RESERVATIONS (confidence: {page_result.get('confidence', 0):.3f})")
            else:
                print(f"📄 PAGE {current_page}: No reservations (confidence: {page_result.get('confidence', 0):.3f})")
            
            # Update progress
            elapsed_time = time.time() - start_time
            avg_time_per_page = elapsed_time / current_page
            estimated_remaining = avg_time_per_page * (total_pages - current_page)
            
            self._update_progress(
                current_page, 
                total_pages, 
                pages_with_reservations.copy(), 
                elapsed_time, 
                estimated_remaining,
                page_result
            )
            
            # Save progress to GCS after each page
            self._save_progress_to_gcs()
        
        # Create final results - include ALL pages (both positive and negative results)
        results = []
        for page_result in page_results:
            results.append({
                'page_number': page_result['page_number'],
                'has_reservations': page_result.get('has_reservations', False),
                'confidence': page_result.get('confidence', 0.0),
                'reasoning': page_result.get('reasoning', 'No reasoning provided')
            })
        
        return {
            "total_pages": total_pages,
            "pages_with_reservations": len(pages_with_reservations),
            "reservation_pages": pages_with_reservations,
            "results": results,
            "processing_method": "page_by_page"
        }
    
    def _save_progress_to_gcs(self):
        """Save current progress to GCS for resume capability"""
        try:
            import os
            import json
            import base64
            from google.cloud import storage
            
            if not self.job_id or not self.job_results:
                return
            
            # Get current progress data
            progress_data = {
                "job_id": self.job_id,
                "timestamp": time.time(),
                "progress": self.job_results[self.job_id].get("progress", {}),
                "status": "processing"
            }
            
            # Initialize GCS client
            credentials_json = base64.b64decode(os.getenv("GOOGLE_CREDENTIALS_BASE64", "")).decode('utf-8')
            credentials = json.loads(credentials_json)
            client = storage.Client.from_service_account_info(credentials)
            bucket_name = os.getenv("GCS_BUCKET_NAME", "mineral-rights-pdfs-1759435410")
            bucket = client.bucket(bucket_name)
            
            # Save progress to GCS
            progress_blob_name = f"progress/{self.job_id}.json"
            progress_blob = bucket.blob(progress_blob_name)
            progress_blob.upload_from_string(
                json.dumps(progress_data, indent=2),
                content_type='application/json'
            )
            
            print(f"💾 Progress saved to GCS: {progress_blob_name}")
            
        except Exception as e:
            print(f"❌ Error saving progress to GCS: {e}")
            # Don't fail the processing if saving progress fails
    
    def process_large_pdf_from_gcs_with_progress(self, gcs_url: str, job_id: str, job_results: dict) -> Dict[str, Any]:
        """
        Process large PDF from GCS with real-time progress updates
        
        Args:
            gcs_url: GCS URL of the PDF file
            job_id: Job ID for progress tracking
            job_results: Global job results dictionary
            
        Returns:
            Dict with pages containing mineral rights reservations
        """
        self.job_id = job_id
        self.job_results = job_results
        
        print(f"🔍 Processing large PDF from GCS with progress tracking: {gcs_url}")
        
        # Download PDF from GCS to temporary file
        import tempfile
        import requests
        
        try:
            # Download PDF from GCS
            print(f"📥 Downloading PDF from GCS...")
            response = requests.get(gcs_url, stream=True)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            print(f"✅ PDF downloaded to temporary file: {tmp_file_path}")
            
            # Process the downloaded file with progress tracking
            result = self.process_large_pdf_with_progress(tmp_file_path, job_id, job_results)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing GCS file: {e}")
            raise
        finally:
            # Clean up temporary file
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                print(f"🧹 Cleaned up temporary file: {tmp_file_path}")
    
    def _process_single_page_with_progress(self, pdf_path: str, page_num: int, current_page: int) -> Dict[str, Any]:
        """Process a single page and return result"""
        try:
            # Use the existing processor's single page method
            # Extract single page from PDF and process it
            import fitz
            from PIL import Image
            from io import BytesIO
            import base64
            
            # Open PDF and get the specific page
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # 2x zoom for quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(BytesIO(img_data))
            
            # Extract text from page using Claude (for image-based PDFs)
            page_text = self.processor.extract_text_with_claude(image, max_tokens=6000)
            doc.close()
            
            # Use the real AI classifier to process this page
            try:
                # Use the real classifier with page-specific settings
                # For individual pages, we need to be more conservative since pages lack full document context
                result = self.processor.classifier.classify_document(
                    ocr_text=page_text,
                    max_samples=1,  # Single sample for speed
                    confidence_threshold=0.9,  # Very high threshold for pages (was 0.6)
                    high_recall_mode=False  # Disable high recall mode for pages (was True)
                )
                
                has_reservations = result.predicted_class == 1
                confidence = result.confidence
                
                # Use the actual AI reasoning from the classification result
                ai_reasoning = "No reasoning provided"
                if result.all_samples and len(result.all_samples) > 0:
                    ai_reasoning = result.all_samples[0].reasoning
                
                return {
                    'page_number': current_page,
                    'has_reservations': has_reservations,
                    'confidence': confidence,
                    'reasoning': ai_reasoning
                }
                
            except Exception as e:
                print(f"❌ Error with AI classification for page {current_page}: {e}")
                # Fallback to simple keyword detection if AI fails
                has_reservations = any(keyword in page_text.lower() for keyword in [
                    'mineral', 'oil', 'gas', 'reservation', 'reserved', 'subsurface'
                ])
                confidence = 0.6 if has_reservations else 0.4
                
                return {
                    'page_number': current_page,
                    'has_reservations': has_reservations,
                    'confidence': confidence,
                    'reasoning': f"Fallback analysis for page {current_page}: {'Found mineral rights keywords' if has_reservations else 'No mineral rights keywords found'} (AI failed: {str(e)})"
                }
            
        except Exception as e:
            print(f"❌ Error processing page {current_page}: {e}")
            return {
                'page_number': current_page,
                'has_reservations': False,
                'confidence': 0.0,
                'reasoning': f"Error processing page: {str(e)}"
            }
    
    def _update_progress(self, current_page: int, total_pages: int, pages_with_reservations: list, 
                        processing_time: float, estimated_remaining: float, current_page_result: dict = None):
        """Update job progress in real-time"""
        if self.job_id and self.job_results and self.job_id in self.job_results:
            self.job_results[self.job_id]["progress"] = {
                "current_page": current_page,
                "total_pages": total_pages,
                "pages_with_reservations": pages_with_reservations,
                "processing_time": processing_time,
                "estimated_remaining": estimated_remaining,
                "current_page_result": current_page_result,
                "progress_percentage": (current_page / total_pages * 100) if total_pages > 0 else 0
            }
            self.job_results[self.job_id]["timestamp"] = time.time()
