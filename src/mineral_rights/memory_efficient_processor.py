import os
import csv
import tempfile
import fitz
import gc
import psutil
from typing import Dict, List, Any, Iterator
from .document_classifier import DocumentProcessor

class MemoryEfficientProcessor:
    """
    Memory-efficient PDF processor that processes one page at a time
    and immediately saves results to avoid memory accumulation.
    """
    
    def __init__(self, api_key: str):
        self.processor = DocumentProcessor(api_key=api_key)
        self.classifier = self.processor.classifier
    
    def process_pdf_streaming(self, pdf_path: str, output_csv: str = None) -> Dict[str, Any]:
        """
        Process PDF page by page with true memory efficiency:
        1. Process one page
        2. Save result immediately 
        3. Clear memory
        4. Move to next page
        """
        print(f"🔍 Processing PDF with streaming approach: {pdf_path}")
        
        # Get total pages first
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()  # Close immediately
        
        print(f"📄 PDF has {total_pages} pages")
        print(f"💾 Starting memory: {self._get_memory_mb():.1f} MB")
        
        # Track results
        pages_with_reservations = []
        total_processing_time = 0
        
        # Open CSV file for streaming results
        csv_file = None
        csv_writer = None
        if output_csv:
            csv_file = open(output_csv, 'w', newline='', encoding='utf-8')
            csv_writer = csv.DictWriter(csv_file, fieldnames=[
                'page_number', 'has_reservations', 'confidence', 'reasoning', 'text_preview'
            ])
            csv_writer.writeheader()
        
        try:
            # Process each page individually
            for page_num in range(total_pages):
                current_page = page_num + 1
                print(f"\n--- PROCESSING PAGE {current_page}/{total_pages} ---")
                
                # Process single page
                result = self._process_single_page(pdf_path, page_num, current_page)
                
                # Track results
                if result['has_reservations']:
                    pages_with_reservations.append(current_page)
                    print(f"🎯 PAGE {current_page}: HAS RESERVATIONS (confidence: {result['confidence']:.3f})")
                else:
                    print(f"📄 PAGE {current_page}: No reservations (confidence: {result['confidence']:.3f})")
                
                # Save result immediately to CSV
                if csv_writer:
                    csv_writer.writerow({
                        'page_number': current_page,
                        'has_reservations': result['has_reservations'],
                        'confidence': result['confidence'],
                        'reasoning': result['reasoning'][:200] + "..." if len(result['reasoning']) > 200 else result['reasoning'],
                        'text_preview': result['text_preview'][:200] + "..." if len(result['text_preview']) > 200 else result['text_preview']
                    })
                    csv_file.flush()  # Force write to disk
                
                # Force memory cleanup after each page
                self._force_cleanup()
                
                # Progress update
                if current_page % 10 == 0:
                    print(f"📊 Progress: {current_page}/{total_pages} pages processed")
                    print(f"💾 Memory: {self._get_memory_mb():.1f} MB")
        
        finally:
            if csv_file:
                csv_file.close()
        
        # Create summary
        summary = {
            'total_pages': total_pages,
            'pages_with_reservations': len(pages_with_reservations),
            'reservation_pages': pages_with_reservations,
            'processing_method': 'streaming_memory_efficient',
            'output_csv': output_csv
        }
        
        print(f"✅ Streaming processing complete: {len(pages_with_reservations)} pages with reservations")
        print(f"💾 Final memory: {self._get_memory_mb():.1f} MB")
        
        return summary
    
    def _process_single_page(self, pdf_path: str, page_num: int, current_page: int) -> Dict[str, Any]:
        """Process a single page and return result immediately"""
        doc = None
        page = None
        pix = None
        image = None
        
        try:
            # Open PDF, process single page, close immediately
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            # Convert to image
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Extract text
            from PIL import Image
            from io import BytesIO
            image = Image.open(BytesIO(img_data))
            page_text = self.processor.extract_text_with_claude(image, max_tokens=6000)
            
            # Classify page
            classification_result = self.classifier.classify_document(
                page_text, 
                max_samples=2,  # Reduced for memory efficiency
                confidence_threshold=0.6,
                high_recall_mode=True
            )
            
            # Get reasoning
            reasoning = (
                classification_result.all_samples[0].reasoning
                if classification_result.all_samples else "No reasoning available"
            )
            
            return {
                'page_number': current_page,
                'has_reservations': classification_result.predicted_class == 1,
                'confidence': classification_result.confidence,
                'reasoning': reasoning,
                'text_preview': page_text[:200] + "..." if len(page_text) > 200 else page_text
            }
            
        finally:
            # CRITICAL: Clean up all objects immediately
            if image:
                image.close()
            if doc:
                doc.close()
            
            # Force cleanup
            del doc, page, pix, image
            gc.collect()
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        gc.collect()  # Call twice for thorough cleanup
