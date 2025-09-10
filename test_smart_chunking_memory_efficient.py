#!/usr/bin/env python3
"""
Memory-efficient test for smart chunking
"""

import os
import json
import time
import fitz
import gc
from typing import List, Tuple, Dict, Any
from google.cloud import documentai
from google.api_core import client_options

def create_smart_chunks(pdf_path: str, max_chunk_size: int = 25, overlap: int = 5) -> List[Tuple[int, int]]:
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

def extract_and_process_chunk(pdf_path: str, start_page: int, end_page: int, chunk_id: int, client, processor_name: str):
    """Extract and process a single chunk, cleaning up immediately"""
    print(f"📤 Processing chunk {chunk_id} (pages {start_page}-{end_page})...")
    
    # Extract chunk
    doc = fitz.open(pdf_path)
    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
    chunk_bytes = chunk_doc.write()
    
    # Clean up immediately
    doc.close()
    chunk_doc.close()
    del doc, chunk_doc
    gc.collect()
    
    # Process chunk
    start_time = time.time()
    
    raw_document = documentai.RawDocument(
        content=chunk_bytes,
        mime_type="application/pdf"
    )
    
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document
    )
    
    try:
        result = client.process_document(request=request)
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
        
        print(f"   ✅ Completed in {processing_time:.2f}s - {len(entities)} entities")
        return entities, processing_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return [], time.time() - start_time

def test_smart_chunking_memory_efficient(pdf_name: str):
    """Test smart chunking with memory efficiency"""
    print(f"🧪 Memory-Efficient Smart Chunking: {pdf_name}")
    print("=" * 50)
    
    pdf_path = f"data/multi-deed/pdfs/{pdf_name}"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Initialize Document AI client
    opts = client_options.ClientOptions(api_endpoint="us-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
    
    # Create smart chunks
    chunks = create_smart_chunks(pdf_path, max_chunk_size=25, overlap=5)
    print(f"📄 Created {len(chunks)} chunks:")
    for i, (start, end) in enumerate(chunks):
        print(f"   Chunk {i+1}: Pages {start}-{end} ({end-start+1} pages)")
    
    # Process chunks one by one
    all_entities = []
    total_processing_time = 0
    
    for i, (start_page, end_page) in enumerate(chunks):
        entities, processing_time = extract_and_process_chunk(
            pdf_path, start_page, end_page, i+1, client, processor_name
        )
        
        # Add chunk metadata
        for entity in entities:
            entity['chunk_id'] = i+1
            entity['chunk_pages'] = f"{start_page}-{end_page}"
        
        all_entities.extend(entities)
        total_processing_time += processing_time
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # Analyze results
    deeds = [e for e in all_entities if e['type'] == 'DEED']
    covers = [e for e in all_entities if e['type'] == 'COVER']
    
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
    
    # Load ground truth for comparison
    pdf_name_no_ext = pdf_name.replace('.pdf', '')
    gt_path = f"data/multi-deed/normalized_boundaries/{pdf_name_no_ext}.normalized.json"
    
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
    
    # Print summary
    print(f"\n📊 Smart Chunking Results:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Total processing time: {total_processing_time:.2f}s")
    print(f"   - Total entities: {len(all_entities)}")
    print(f"   - DEED entities: {len(deeds)}")
    print(f"   - COVER entities: {len(covers)}")
    print(f"   - Unique deed ranges: {len(unique_deed_ranges)}")
    
    if deed_count > 0:
        print(f"   - Ground truth deeds: {deed_count}")
        print(f"   - Over-detection ratio: {over_detection_ratio:.2f}x")
        print(f"   - Systematic offset: {systematic_offset}")
        if systematic_offset:
            print(f"   - Offset value: {mean_offset:.1f}")
    
    # Show first few deed ranges
    print(f"\n📄 First 5 Deed Ranges:")
    for i, (start, end) in enumerate(unique_deed_ranges[:5]):
        print(f"   Deed {i+1}: Pages {start}-{end}")

if __name__ == "__main__":
    import sys
    pdf_name = sys.argv[1] if len(sys.argv) > 1 else "FRANCO.pdf"
    test_smart_chunking_memory_efficient(pdf_name)

