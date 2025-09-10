#!/usr/bin/env python3
"""
Final Smart Chunking Implementation - 15-page chunks with overlap
"""

import os
import json
import time
import fitz
import gc
from typing import List, Tuple, Dict, Any
from google.cloud import documentai
from google.api_core import client_options

def create_smart_chunks_15(pdf_path: str, overlap: int = 3) -> List[Tuple[int, int]]:
    """Create smart chunks with 15-page limit and overlap"""
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
        # Calculate max end page (15-page limit)
        max_end = min(start + 14, total_pages - 1)  # 0-indexed, so 14 = 15 pages
        
        # Try to find a good boundary within the chunk
        best_end = start + 14  # Default to max chunk size
        
        # Look for deed boundaries within the chunk
        for boundary in deed_boundaries:
            if start < boundary <= max_end:
                # End just before the next deed boundary
                best_end = boundary - 1
                break
        
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

def merge_overlapping_deeds(all_entities: List[Dict]) -> List[Dict]:
    """Merge overlapping deed detections"""
    deeds = [e for e in all_entities if e['type'] == 'DEED']
    
    if not deeds:
        return []
    
    # Sort by starting page
    deeds.sort(key=lambda x: min([int(ref['page']) for ref in x['page_refs']]) if x['page_refs'] else 0)
    
    merged_deeds = []
    current_deed = None
    
    for deed in deeds:
        if not deed['page_refs']:
            continue
            
        deed_pages = [int(ref['page']) for ref in deed['page_refs']]
        deed_start = min(deed_pages)
        deed_end = max(deed_pages)
        
        if current_deed is None:
            current_deed = {
                'type': 'DEED',
                'confidence': deed['confidence'],
                'page_refs': deed['page_refs'].copy(),
                'pages': deed_pages.copy()
            }
        else:
            current_pages = [int(ref['page']) for ref in current_deed['page_refs']]
            current_start = min(current_pages)
            current_end = max(current_pages)
            
            # Check for overlap or adjacency (within 2 pages)
            if deed_start <= current_end + 2:
                # Merge
                current_deed['pages'].extend(deed_pages)
                current_deed['pages'] = sorted(list(set(current_deed['pages'])))
                current_deed['confidence'] = max(current_deed['confidence'], deed['confidence'])
                
                # Update page_refs
                current_deed['page_refs'] = [{'page': str(p), 'confidence': current_deed['confidence']} for p in current_deed['pages']]
            else:
                # No overlap, finalize current deed and start new one
                merged_deeds.append(current_deed)
                current_deed = {
                    'type': 'DEED',
                    'confidence': deed['confidence'],
                    'page_refs': deed['page_refs'].copy(),
                    'pages': deed_pages.copy()
                }
    
    if current_deed:
        merged_deeds.append(current_deed)
    
    return merged_deeds

def test_smart_chunking_15_pages(pdf_name: str):
    """Test smart chunking with 15-page chunks"""
    print(f"🧪 Smart Chunking (15-page chunks): {pdf_name}")
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
    chunks = create_smart_chunks_15(pdf_path, overlap=3)
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
    
    # Merge overlapping deeds
    merged_deeds = merge_overlapping_deeds(all_entities)
    
    # Analyze results
    covers = [e for e in all_entities if e['type'] == 'COVER']
    
    # Get unique deed ranges
    unique_deed_ranges = []
    for deed in merged_deeds:
        if deed['pages']:
            start_page = min(deed['pages'])
            end_page = max(deed['pages'])
            unique_deed_ranges.append((start_page, end_page))
    
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
    print(f"   - DEED entities (before merge): {len([e for e in all_entities if e['type'] == 'DEED'])}")
    print(f"   - DEED entities (after merge): {len(merged_deeds)}")
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
    
    return {
        'chunks': len(chunks),
        'processing_time': total_processing_time,
        'deed_count': len(unique_deed_ranges),
        'over_detection_ratio': over_detection_ratio,
        'systematic_offset': systematic_offset,
        'mean_offset': mean_offset
    }

if __name__ == "__main__":
    import sys
    pdf_name = sys.argv[1] if len(sys.argv) > 1 else "FRANCO.pdf"
    test_smart_chunking_15_pages(pdf_name)
