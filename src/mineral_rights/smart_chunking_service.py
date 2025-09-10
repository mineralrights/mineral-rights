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
        
        # Initialize Document AI client
        opts = client_options.ClientOptions(api_endpoint="us-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
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
        """Process a single chunk and return deed detections"""
        # Extract chunk
        doc = fitz.open(pdf_path)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
        chunk_bytes = chunk_doc.write()
        
        # Clean up
        doc.close()
        chunk_doc.close()
        gc.collect()
        
        # Process chunk
        raw_document = documentai.RawDocument(
            content=chunk_bytes,
            mime_type="application/pdf"
        )
        
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        
        try:
            result = self.client.process_document(request=request)
            
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
            del result, raw_document, request, chunk_bytes
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
        
        # Create smart chunks
        chunks = self.create_smart_chunks(pdf_path, chunk_size, overlap)
        
        # Process chunks one by one
        all_deeds = []
        
        for i, (start_page, end_page) in enumerate(chunks):
            chunk_id = i + 1
            chunk_deeds = self.process_chunk(pdf_path, start_page, end_page, chunk_id)
            all_deeds.extend(chunk_deeds)
            
            # Force garbage collection after each chunk
            gc.collect()
        
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

def create_smart_chunking_service(project_id: str, location: str, processor_id: str, processor_version: str, credentials_path: Optional[str] = None) -> SmartChunkingService:
    """Factory function to create a SmartChunkingService instance"""
    return SmartChunkingService(project_id, location, processor_id, processor_version, credentials_path)
