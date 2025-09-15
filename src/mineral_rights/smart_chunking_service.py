"""
Smart Chunking Service for Document AI Deed Detection
Production-ready implementation for the Vercel app
"""

import os
import json
import time
import fitz
import gc
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from google.cloud import documentai
from google.api_core import client_options
from google.oauth2 import service_account

@dataclass
class DeedDetectionResult:
    """Result of deed detection for a single deed"""
    deed_number: int
    start_page: int
    end_page: int
    confidence: float
    pages: List[int]

@dataclass
class SmartChunkingResult:
    """Result of smart chunking processing"""
    total_deeds: int
    deed_detections: List[DeedDetectionResult]
    processing_time: float
    chunks_processed: int
    systematic_offset: Optional[int] = None
    raw_deeds_before_merge: int = 0

class SmartChunkingService:
    """Production-ready smart chunking service for Document AI deed detection"""
    
    def __init__(self, project_id: str, location: str, processor_id: str, processor_version: str, credentials_path: Optional[str] = None):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.processor_version = processor_version
        self.credentials_path = credentials_path
        
        # Initialize Document AI client with credentials
        opts = client_options.ClientOptions(api_endpoint="us-documentai.googleapis.com")
        
        if self.credentials_path and os.path.exists(self.credentials_path):
            # Use service account file
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.client = documentai.DocumentProcessorServiceClient(credentials=credentials, client_options=opts)
            print(f"✅ SmartChunkingService initialized with credentials: {self.credentials_path}")
        else:
            # Use default credentials (e.g., from environment)
            self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
            print("✅ SmartChunkingService initialized with default credentials")
            
        self.processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version}"
    
    def create_smart_chunks(self, pdf_path: str, chunk_size: int = 15, overlap: int = 3) -> List[Tuple[int, int]]:
        """Create smart chunks with specified size and overlap"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        chunks = []
        start = 0
        
        while start < total_pages:
            # Calculate end page
            end = min(start + chunk_size - 1, total_pages - 1)  # 0-indexed
            chunks.append((start + 1, end + 1))  # Convert to 1-indexed
            
            # Move start to create overlap
            start = end - overlap + 1
            
            # Ensure we don't go backwards
            if start <= chunks[-1][0]:
                start = chunks[-1][1] - overlap + 1
        
        return chunks
    
    def process_chunk(self, pdf_path: str, start_page: int, end_page: int, chunk_id: int) -> List[Dict[str, Any]]:
        """Process a single chunk and return deed detections with memory optimization"""
        chunk_bytes = None
        doc = None
        chunk_doc = None
        result = None
        
        try:
            # Extract chunk
            doc = fitz.open(pdf_path)
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
            chunk_bytes = chunk_doc.write()
            print(f"✅ Chunk {chunk_id} extracted successfully, size: {len(chunk_bytes)} bytes")
            
            # Process chunk
            raw_document = documentai.RawDocument(
                content=chunk_bytes,
                mime_type="application/pdf"
            )
            
            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document
            )
            
            print(f"📡 Sending chunk {chunk_id} to Document AI...")
            result = self.client.process_document(request=request)
            print(f"✅ Chunk {chunk_id} processed by Document AI successfully")
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            # Clean up resources even on error
            if doc:
                doc.close()
            if chunk_doc:
                chunk_doc.close()
            if 'chunk_bytes' in locals() and chunk_bytes:
                del chunk_bytes
            return []
        finally:
            # Aggressive cleanup
            if doc:
                doc.close()
            if chunk_doc:
                chunk_doc.close()
            if 'chunk_bytes' in locals() and chunk_bytes:
                del chunk_bytes
            # Force garbage collection
            import gc
            gc.collect()
        
        if result is None:
            return []
        
        try:
            
            # Parse entities
            entities = []
            for entity in result.document.entities:
                if entity.type_ == 'DEED':
                    entity_dict = {
                        'type': entity.type_,
                        'confidence': entity.confidence,
                        'pages': [],
                        'chunk_id': chunk_id,
                        'chunk_start': start_page,
                        'chunk_end': end_page
                    }
                    
                    # Extract page references
                    if entity.page_anchor and entity.page_anchor.page_refs:
                        for ref in entity.page_anchor.page_refs:
                            if ref.page is not None and str(ref.page).isdigit():
                                # Adjust page number to original PDF
                                adjusted_page = int(ref.page) + start_page
                                entity_dict['pages'].append(adjusted_page)
                    
                    # Only include deeds with valid pages
                    if entity_dict['pages']:
                        entities.append(entity_dict)
            
            # Clean up
            del result, raw_document, request
            if 'chunk_bytes' in locals():
                del chunk_bytes
            gc.collect()
            
            return entities
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            return []
    
    def merge_deeds(self, all_deeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping deeds with improved logic"""
        if not all_deeds:
            return []
        
        # Sort by starting page
        all_deeds.sort(key=lambda x: min(x['pages']) if x['pages'] else 0)
        
        merged_deeds = []
        
        for deed in all_deeds:
            if not deed['pages']:
                continue
                
            deed_start = min(deed['pages'])
            deed_end = max(deed['pages'])
            
            # Check if this deed should be merged with any existing merged deed
            merged = False
            
            for i, merged_deed in enumerate(merged_deeds):
                merged_start = min(merged_deed['pages'])
                merged_end = max(merged_deed['pages'])
                
                # Check for significant overlap (at least 2 pages)
                overlap_start = max(deed_start, merged_start)
                overlap_end = min(deed_end, merged_end)
                overlap_pages = max(0, overlap_end - overlap_start + 1)
                
                # Merge if there's significant overlap
                if overlap_pages >= 2:
                    # Merge pages
                    merged_pages = sorted(list(set(merged_deed['pages'] + deed['pages'])))
                    merged_deeds[i] = {
                        'type': 'DEED',
                        'confidence': max(merged_deed['confidence'], deed['confidence']),
                        'pages': merged_pages,
                        'chunk_id': f"{merged_deed.get('chunk_id', 'unknown')}+{deed.get('chunk_id', 'unknown')}"
                    }
                    merged = True
                    break
            
            if not merged:
                merged_deeds.append(deed)
        
        return merged_deeds
    
    def apply_offset_correction(self, merged_deeds: List[Dict[str, Any]], offset: int = 1) -> List[Dict[str, Any]]:
        """Apply systematic offset correction"""
        corrected_deeds = []
        
        for deed in merged_deeds:
            corrected_pages = [p + offset for p in deed['pages']]
            corrected_deed = deed.copy()
            corrected_deed['pages'] = corrected_pages
            corrected_deed['original_pages'] = deed['pages']  # Keep original for reference
            corrected_deeds.append(corrected_deed)
        
        return corrected_deeds
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 15, overlap: int = 3, apply_offset: bool = True) -> SmartChunkingResult:
        """Process PDF using smart chunking and return structured results"""
        start_time = time.time()
        
        # Check PDF size and adjust chunk size for memory efficiency
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # TWO-STAGE APPROACH: Use Document AI for all PDFs with optimized memory management
        print(f"🎯 Two-stage approach: Using Document AI for deed boundary detection on {total_pages} pages")
        
        # Dynamic chunk sizing based on PDF size for memory optimization
        if total_pages > 50:
            chunk_size = 2  # Very small chunks for very large PDFs
            print(f"📦 Very large PDF detected ({total_pages} pages), using chunk size {chunk_size}")
        elif total_pages > 30:
            chunk_size = 3  # Small chunks for large PDFs
            print(f"📦 Large PDF detected ({total_pages} pages), using chunk size {chunk_size}")
        elif total_pages > 15:
            chunk_size = 5  # Medium chunks for medium PDFs
            print(f"📦 Medium PDF detected ({total_pages} pages), using chunk size {chunk_size}")
        else:
            chunk_size = min(chunk_size, 8)  # Default chunks for small PDFs
            print(f"📦 Small PDF detected ({total_pages} pages), using chunk size {chunk_size}")
        
        # Create smart chunks
        chunks = self.create_smart_chunks(pdf_path, chunk_size, overlap)
        
        # Process chunks one by one
        all_deeds = []
        
        for i, (start_page, end_page) in enumerate(chunks):
            chunk_id = i + 1
            print(f"📄 Processing chunk {chunk_id}/{len(chunks)} (pages {start_page}-{end_page})")
            
            # Check memory before processing chunk
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            print(f"💾 Memory before chunk {chunk_id}: {memory_before:.1f} MB")
            
            # Skip chunk if memory is too high (32GB = 32000MB, use 28000MB as threshold)
            if memory_before > 28000:  # 28GB threshold for 32GB container
                print(f"⚠️ Memory too high ({memory_before:.1f} MB), skipping chunk {chunk_id}")
                continue
            
            chunk_deeds = self.process_chunk(pdf_path, start_page, end_page, chunk_id)
            all_deeds.extend(chunk_deeds)
            
            # Aggressive memory management after each chunk
            print(f"🧹 Cleaning up memory after chunk {chunk_id}...")
            import gc
            gc.collect()
            
            # Add delay between chunks to allow memory cleanup
            import time as time_module
            print("⏳ Adding delay for memory cleanup...")
            time_module.sleep(3)  # 3 second delay
            
            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"💾 Memory after chunk {chunk_id}: {memory_mb:.1f} MB")
            
            # If memory usage is too high, force more aggressive cleanup
            if memory_mb > 25000:  # 25GB threshold for 32GB container
                print("⚠️ High memory usage detected, forcing aggressive cleanup...")
                import gc
                gc.set_threshold(0)  # Disable automatic garbage collection
                gc.collect()
                gc.set_threshold(700, 10, 10)  # Restore default thresholds
                
                # Add delay between chunks to allow memory cleanup
                import time as time_module
                print("⏳ Adding delay for memory cleanup...")
                time_module.sleep(2)
        
        # Merge overlapping deeds
        merged_deeds = self.merge_deeds(all_deeds)
        
        # Apply offset correction if requested
        if apply_offset:
            final_deeds = self.apply_offset_correction(merged_deeds, offset=1)
            systematic_offset = 1
        else:
            final_deeds = merged_deeds
            systematic_offset = None
        
        # Convert to structured results
        deed_detections = []
        for i, deed in enumerate(final_deeds):
            if deed['pages']:
                deed_detection = DeedDetectionResult(
                    deed_number=i + 1,
                    start_page=min(deed['pages']),
                    end_page=max(deed['pages']),
                    confidence=deed['confidence'],
                    pages=deed['pages']
                )
                deed_detections.append(deed_detection)
        
        processing_time = time.time() - start_time
        
        return SmartChunkingResult(
            total_deeds=len(deed_detections),
            deed_detections=deed_detections,
            processing_time=processing_time,
            chunks_processed=len(chunks),
            systematic_offset=systematic_offset,
            raw_deeds_before_merge=len(all_deeds)
        )
    
    def _fallback_to_simple_splitting(self, pdf_path: str) -> SmartChunkingResult:
        """Fallback to simple page-based splitting for large PDFs to avoid memory issues"""
        import time
        import fitz
        
        start_time = time.time()
        
        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"📄 Using simple page-based splitting for {total_pages} pages")
        
        # Simple strategy: every 3 pages = 1 deed (with overlap)
        pages_per_deed = 3
        overlap = 1
        deeds = []
        
        for start_page in range(0, total_pages, pages_per_deed - overlap):
            end_page = min(start_page + pages_per_deed, total_pages)
            deed_number = len(deeds) + 1
            
            # Create deed detection result
            deed_pages = list(range(start_page + 1, end_page + 1))  # 1-indexed pages
            
            deed_result = DeedDetectionResult(
                deed_number=deed_number,
                start_page=min(deed_pages),
                end_page=max(deed_pages),
                confidence=0.8,  # Default confidence for simple splitting
                pages=deed_pages
            )
            
            deeds.append(deed_result)
            print(f"📄 Created deed {deed_number}: pages {deed_pages}")
        
        processing_time = time.time() - start_time
        
        print(f"✅ Simple splitting completed: {len(deeds)} deeds in {processing_time:.1f}s")
        
        return SmartChunkingResult(
            total_deeds=len(deeds),
            deed_detections=deeds,
            processing_time=processing_time,
            chunks_processed=1,  # Single "chunk" for simple splitting
            systematic_offset=None,
            raw_deeds_before_merge=len(deeds)
        )

def create_smart_chunking_service(project_id: str, location: str, processor_id: str, processor_version: str, credentials_path: Optional[str] = None) -> SmartChunkingService:
    """Factory function to create a SmartChunkingService instance"""
    return SmartChunkingService(project_id, location, processor_id, processor_version, credentials_path)
