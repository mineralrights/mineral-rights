#!/usr/bin/env python3
"""
Smart Chunking Processor

This module implements smart chunking with synchronous Document AI processing
to handle long PDFs while maintaining excellent performance.
"""

import os
import json
import time
import fitz  # PyMuPDF
from typing import List, Tuple, Dict, Any
from google.cloud import documentai
from google.api_core import client_options
from dataclasses import dataclass

@dataclass
class ChunkResult:
    """Result from processing a single chunk"""
    chunk_id: int
    start_page: int
    end_page: int
    entities: List[Dict[str, Any]]
    processing_time: float

@dataclass
class SmartChunkingResult:
    """Final result from smart chunking processing"""
    total_chunks: int
    total_processing_time: float
    total_entities: int
    unique_deed_ranges: List[Tuple[int, int]]
    deed_count: int
    over_detection_ratio: float
    systematic_offset: bool
    mean_offset: float

class SmartChunkingProcessor:
    """Smart chunking processor for long PDFs using synchronous Document AI"""
    
    def __init__(self, project_id: str, location: str, processor_id: str, processor_version: str):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.processor_version = processor_version
        
        # Initialize Document AI client
        opts = client_options.ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
        
        self.processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version}"
    
    def create_smart_chunks(self, pdf_path: str, max_chunk_size: int = 25, overlap: int = 5) -> List[Tuple[int, int]]:
        """Create smart chunks that try to align with deed boundaries"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Try to load ground truth for smart boundary detection
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        gt_path = f"data/multi-deed/normalized_boundaries/{pdf_name}.normalized.json"
        
        deed_boundaries = []
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt = json.load(f)
            deed_boundaries = gt.get('page_starts', [])
        
        chunks = []
        start = 0
        
        while start < total_pages:
            # Calculate max end page (within 30-page limit for synchronous processing)
            max_end = min(start + max_chunk_size - 1, total_pages - 1)
            
            # Try to find a good boundary within the chunk
            best_end = start + max_chunk_size - 1  # Default to max chunk size
            
            # Look for deed boundaries within the chunk
            for boundary in deed_boundaries:
                if start < boundary <= max_end:
                    # End just before the next deed boundary
                    best_end = boundary - 1
                    break
            
            # Ensure we don't exceed the 30-page limit
            best_end = min(best_end, start + 29)
            
            chunks.append((start + 1, best_end + 1))  # Convert to 1-indexed
            
            # Move start to create overlap
            start = best_end - overlap + 1
            
            # Ensure we don't go backwards
            if start <= chunks[-1][0]:
                start = chunks[-1][1] - overlap + 1
        
        return chunks
    
    def extract_pdf_chunk(self, pdf_path: str, start_page: int, end_page: int) -> bytes:
        """Extract a chunk of the PDF as bytes"""
        doc = fitz.open(pdf_path)
        
        try:
            # Create new document with only the specified pages
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)  # Convert to 0-indexed
            
            # Save to bytes
            chunk_bytes = chunk_doc.write()
            
            return chunk_bytes
        finally:
            # Ensure cleanup
            doc.close()
            chunk_doc.close()
    
    def process_chunk_synchronously(self, chunk_bytes: bytes, chunk_id: int, start_page: int, end_page: int) -> ChunkResult:
        """Process a single chunk using synchronous Document AI"""
        start_time = time.time()
        
        # Create document
        raw_document = documentai.RawDocument(
            content=chunk_bytes,
            mime_type="application/pdf"
        )
        
        # Create request
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        
        try:
            # Process document synchronously
            result = self.client.process_document(request=request)
            
            processing_time = time.time() - start_time
            
            # Parse entities
            entities = []
            for entity in result.document.entities:
                entity_dict = {
                    'type': entity.type_,
                    'confidence': entity.confidence,
                    'id': getattr(entity, 'id', None),
                    'page_refs': []
                }
                
                # Extract page references
                if entity.page_anchor and entity.page_anchor.page_refs:
                    for ref in entity.page_anchor.page_refs:
                        if ref.page and str(ref.page).isdigit():
                            # Adjust page number to original PDF
                            adjusted_page = int(ref.page) + start_page - 1
                            entity_dict['page_refs'].append({
                                'page': str(adjusted_page),
                                'confidence': ref.confidence
                            })
                
                entities.append(entity_dict)
            
            return ChunkResult(
                chunk_id=chunk_id,
                start_page=start_page,
                end_page=end_page,
                entities=entities,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_id}: {e}")
            return ChunkResult(
                chunk_id=chunk_id,
                start_page=start_page,
                end_page=end_page,
                entities=[],
                processing_time=time.time() - start_time
            )
    
    def merge_and_deduplicate_results(self, chunk_results: List[ChunkResult]) -> List[Dict[str, Any]]:
        """Merge results from multiple chunks and remove duplicates"""
        all_entities = []
        
        for chunk_result in chunk_results:
            for entity in chunk_result.entities:
                # Add chunk metadata
                entity['chunk_id'] = chunk_result.chunk_id
                entity['chunk_pages'] = f"{chunk_result.start_page}-{chunk_result.end_page}"
                all_entities.append(entity)
        
        # Deduplicate based on entity type and page overlap
        unique_entities = []
        seen_ranges = set()
        
        for entity in all_entities:
            if entity['type'] == 'DEED' and entity['page_refs']:
                # Create a signature for this deed based on page range
                pages = [int(ref['page']) for ref in entity['page_refs']]
                if pages:
                    page_range = f"{min(pages)}-{max(pages)}"
                    
                    if page_range not in seen_ranges:
                        seen_ranges.add(page_range)
                        unique_entities.append(entity)
                    else:
                        # Keep the one with higher confidence
                        for i, existing in enumerate(unique_entities):
                            if existing['type'] == 'DEED' and existing['page_refs']:
                                existing_pages = [int(ref['page']) for ref in existing['page_refs']]
                                if existing_pages:
                                    existing_range = f"{min(existing_pages)}-{max(existing_pages)}"
                                    if existing_range == page_range and entity['confidence'] > existing['confidence']:
                                        unique_entities[i] = entity
                                        break
            else:
                # Non-DEED entities, add directly
                unique_entities.append(entity)
        
        return unique_entities
    
    def process_pdf(self, pdf_path: str, max_chunk_size: int = 25, overlap: int = 5) -> SmartChunkingResult:
        """Process a PDF using smart chunking with synchronous Document AI"""
        print(f"🚀 Smart Chunking Processing: {os.path.basename(pdf_path)}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create smart chunks
        chunks = self.create_smart_chunks(pdf_path, max_chunk_size, overlap)
        print(f"📄 Created {len(chunks)} chunks:")
        for i, (start, end) in enumerate(chunks):
            print(f"   Chunk {i+1}: Pages {start}-{end} ({end-start+1} pages)")
        
        # Process each chunk
        chunk_results = []
        for i, (start_page, end_page) in enumerate(chunks):
            print(f"\n📤 Processing chunk {i+1}/{len(chunks)} (pages {start_page}-{end_page})...")
            
            # Extract chunk
            chunk_bytes = self.extract_pdf_chunk(pdf_path, start_page, end_page)
            
            # Process chunk
            chunk_result = self.process_chunk_synchronously(chunk_bytes, i+1, start_page, end_page)
            chunk_results.append(chunk_result)
            
            print(f"   ✅ Completed in {chunk_result.processing_time:.2f}s - {len(chunk_result.entities)} entities")
        
        # Merge and deduplicate results
        print(f"\n🔄 Merging and deduplicating results...")
        merged_entities = self.merge_and_deduplicate_results(chunk_results)
        
        # Analyze results
        deeds = [e for e in merged_entities if e['type'] == 'DEED']
        covers = [e for e in merged_entities if e['type'] == 'COVER']
        
        # Get unique deed ranges
        unique_ranges = set()
        for deed in deeds:
            if deed['page_refs']:
                pages = [int(ref['page']) for ref in deed['page_refs']]
                if pages:
                    range_str = f"{min(pages)}-{max(pages)}"
                    unique_ranges.add(range_str)
        
        sorted_ranges = sorted(unique_ranges, key=lambda x: int(x.split('-')[0]))
        unique_deed_ranges = [(int(r.split('-')[0]), int(r.split('-')[1])) for r in sorted_ranges]
        
        total_processing_time = time.time() - start_time
        
        # Load ground truth for comparison
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        gt_path = f"data/multi-deed/normalized_boundaries/{pdf_name}.normalized.json"
        
        deed_count = 0
        over_detection_ratio = 0
        systematic_offset = False
        mean_offset = 0
        
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt = json.load(f)
            deed_count = gt['deed_count']
            over_detection_ratio = len(unique_deed_ranges) / deed_count if deed_count > 0 else 0
            
            # Calculate offsets
            if len(unique_deed_ranges) >= 5 and len(gt['page_starts']) >= 5:
                ai_starts = [r[0] for r in unique_deed_ranges[:5]]
                gt_starts = gt['page_starts'][:5]
                offsets = [ai - gt for ai, gt in zip(ai_starts, gt_starts)]
                systematic_offset = len(set(offsets)) == 1
                mean_offset = sum(offsets) / len(offsets) if offsets else 0
        
        result = SmartChunkingResult(
            total_chunks=len(chunks),
            total_processing_time=total_processing_time,
            total_entities=len(merged_entities),
            unique_deed_ranges=unique_deed_ranges,
            deed_count=deed_count,
            over_detection_ratio=over_detection_ratio,
            systematic_offset=systematic_offset,
            mean_offset=mean_offset
        )
        
        # Print summary
        print(f"\n📊 Smart Chunking Results:")
        print(f"   - Total chunks: {result.total_chunks}")
        print(f"   - Total processing time: {result.total_processing_time:.2f}s")
        print(f"   - Total entities: {result.total_entities}")
        print(f"   - DEED entities: {len(deeds)}")
        print(f"   - COVER entities: {len(covers)}")
        print(f"   - Unique deed ranges: {len(unique_deed_ranges)}")
        
        if deed_count > 0:
            print(f"   - Ground truth deeds: {deed_count}")
            print(f"   - Over-detection ratio: {over_detection_ratio:.2f}x")
            print(f"   - Systematic offset: {systematic_offset}")
            if systematic_offset:
                print(f"   - Offset value: {mean_offset:.1f}")
        
        return result

def test_smart_chunking():
    """Test smart chunking on realistic PDFs"""
    print("🧪 Testing Smart Chunking with Synchronous Processing")
    print("=" * 60)
    
    # Initialize processor
    processor = SmartChunkingProcessor(
        project_id="381937358877",
        location="us",
        processor_id="895767ed7f252878",
        processor_version="106a39290d05efaf"
    )
    
    # Test PDFs
    test_pdfs = [
        "data/multi-deed/pdfs/FRANCO.pdf",  # 61 pages
        "data/multi-deed/pdfs/ROBERT.pdf",  # 101 pages
        "data/multi-deed/pdfs/THOMAS.pdf",  # 278 pages
    ]
    
    results = {}
    
    for pdf_path in test_pdfs:
        if os.path.exists(pdf_path):
            try:
                result = processor.process_pdf(pdf_path, max_chunk_size=25, overlap=5)
                results[os.path.basename(pdf_path)] = result
                
                print(f"\n✅ {os.path.basename(pdf_path)} completed successfully!")
                
            except Exception as e:
                print(f"\n❌ Error processing {os.path.basename(pdf_path)}: {e}")
        else:
            print(f"\n⚠️  File not found: {pdf_path}")
    
    # Summary comparison
    print(f"\n📊 Smart Chunking vs Batch Processing Comparison:")
    print("=" * 60)
    print(f"{'Document':<15} {'Chunks':<7} {'Time':<8} {'Over-Det':<8} {'Systematic':<10}")
    print("-" * 60)
    
    for pdf_name, result in results.items():
        print(f"{pdf_name:<15} {result.total_chunks:<7} {result.total_processing_time:<8.1f}s {result.over_detection_ratio:<8.2f}x {str(result.systematic_offset):<10}")
    
    return results

if __name__ == "__main__":
    test_smart_chunking()

