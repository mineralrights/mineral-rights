#!/usr/bin/env python3
"""
Complete Smart Chunking Implementation with offset correction
"""

import os
import json
import time
import fitz
import gc
from typing import List, Tuple, Dict, Any
from google.cloud import documentai
from google.api_core import client_options

def create_smart_chunks(pdf_path: str, overlap: int = 3) -> List[Tuple[int, int]]:
    """Create smart chunks with 15-page limit and overlap"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    chunks = []
    start = 0
    
    while start < total_pages:
        # Calculate max end page (15-page limit)
        max_end = min(start + 14, total_pages - 1)  # 0-indexed, so 14 = 15 pages
        
        chunks.append((start + 1, max_end + 1))  # Convert to 1-indexed
        
        # Move start to create overlap
        start = max_end - overlap + 1
        
        # Ensure we don't go backwards
        if start <= chunks[-1][0]:
            start = chunks[-1][1] - overlap + 1
    
    return chunks

def process_chunk(pdf_path: str, start_page: int, end_page: int, chunk_id: int, client, processor_name: str):
    """Process a single chunk"""
    print(f"📤 Processing chunk {chunk_id} (pages {start_page}-{end_page})...")
    
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
            if entity.type_ == 'DEED':
                entity_dict = {
                    'type': entity.type_,
                    'confidence': entity.confidence,
                    'pages': [],
                    'chunk_id': chunk_id,
                    'chunk_start': start_page
                }
                
                # Extract page references
                if entity.page_anchor and entity.page_anchor.page_refs:
                    for ref in entity.page_anchor.page_refs:
                        if ref.page and str(ref.page).isdigit():
                            # Adjust page number to original PDF
                            adjusted_page = int(ref.page) + start_page - 1
                            entity_dict['pages'].append(adjusted_page)
                
                # Only include deeds with valid pages
                if entity_dict['pages']:
                    entities.append(entity_dict)
        
        print(f"   ✅ Completed in {processing_time:.2f}s - {len(entities)} deeds")
        return entities, processing_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return [], time.time() - start_time

def merge_deeds(all_deeds):
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
            
            # Merge if there's significant overlap (at least 2 pages)
            if overlap_pages >= 2:
                # Merge pages
                merged_pages = sorted(list(set(merged_deed['pages'] + deed['pages'])))
                merged_deeds[i] = {
                    'type': 'DEED',
                    'confidence': max(merged_deed['confidence'], deed['confidence']),
                    'pages': merged_pages,
                    'chunk_id': f"{merged_deed.get('chunk_id', 'unknown')}+{deed.get('chunk_id', 'unknown')}",
                    'chunk_start': min(merged_deed.get('chunk_start', 999), deed.get('chunk_start', 999))
                }
                merged = True
                break
        
        if not merged:
            merged_deeds.append(deed)
    
    return merged_deeds

def apply_offset_correction(merged_deeds, offset: int = 1):
    """Apply systematic offset correction"""
    corrected_deeds = []
    
    for deed in merged_deeds:
        corrected_pages = [p + offset for p in deed['pages']]
        corrected_deed = deed.copy()
        corrected_deed['pages'] = corrected_pages
        corrected_deed['original_pages'] = deed['pages']  # Keep original for reference
        corrected_deeds.append(corrected_deed)
    
    return corrected_deeds

def test_complete_smart_chunking(pdf_name: str, apply_offset: bool = True):
    """Test complete smart chunking on a PDF"""
    print(f"🧪 Complete Smart Chunking: {pdf_name}")
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
    chunks = create_smart_chunks(pdf_path, overlap=3)
    print(f"📄 Created {len(chunks)} chunks:")
    for i, (start, end) in enumerate(chunks):
        print(f"   Chunk {i+1}: Pages {start}-{end} ({end-start+1} pages)")
    
    # Process chunks one by one
    all_deeds = []
    total_time = 0
    
    for i, (start_page, end_page) in enumerate(chunks):
        deeds, processing_time = process_chunk(
            pdf_path, start_page, end_page, i+1, client, processor_name
        )
        
        all_deeds.extend(deeds)
        total_time += processing_time
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # Merge overlapping deeds
    merged_deeds = merge_deeds(all_deeds)
    
    # Apply offset correction if requested
    if apply_offset:
        print(f"\n🔧 Applying +1 page offset correction...")
        final_deeds = apply_offset_correction(merged_deeds, offset=1)
    else:
        final_deeds = merged_deeds
    
    # Analyze results
    print(f"\n📊 Smart Chunking Results:")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Total processing time: {total_time:.2f}s")
    print(f"   - Deeds before merge: {len(all_deeds)}")
    print(f"   - Deeds after merge: {len(merged_deeds)}")
    print(f"   - Final deeds: {len(final_deeds)}")
    
    # Load ground truth for comparison
    pdf_name_no_ext = pdf_name.replace('.pdf', '')
    gt_path = f"data/multi-deed/normalized_boundaries/{pdf_name_no_ext}.normalized.json"
    
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        
        gt_deed_count = gt['deed_count']
        gt_page_starts = gt['page_starts']
        
        print(f"\n📈 Comparison with Ground Truth:")
        print(f"   - Ground truth deeds: {gt_deed_count}")
        print(f"   - Detected deeds: {len(final_deeds)}")
        print(f"   - Detection ratio: {len(final_deeds)/gt_deed_count:.2f}x")
        
        # Calculate accuracy metrics
        ai_starts = [min(d['pages']) for d in final_deeds if d['pages']]
        if ai_starts and gt_page_starts:
            # Check matches within 1 page tolerance
            matches = 0
            for gt_start in gt_page_starts:
                closest_ai_start = min(ai_starts, key=lambda x: abs(x - gt_start))
                if abs(closest_ai_start - gt_start) <= 1:
                    matches += 1
            
            print(f"   - Matches (within 1 page): {matches}/{gt_deed_count}")
            print(f"   - Match rate: {matches/gt_deed_count:.2f}")
        
        # Show first few deed ranges
        print(f"\n📄 First 5 Detected Deeds:")
        for i, deed in enumerate(final_deeds[:5]):
            pages = deed['pages']
            if pages:
                print(f"   Deed {i+1}: Pages {min(pages)}-{max(pages)} (Confidence: {deed['confidence']:.3f})")
        
        # Show ground truth for comparison
        print(f"\n📄 Ground Truth Deed Starts:")
        for i, start in enumerate(gt_page_starts[:5]):
            print(f"   Deed {i+1}: Page {start}")
    
    return {
        'chunks': len(chunks),
        'processing_time': total_time,
        'deed_count': len(final_deeds),
        'deeds': final_deeds
    }

if __name__ == "__main__":
    import sys
    pdf_name = sys.argv[1] if len(sys.argv) > 1 else "FRANCO.pdf"
    apply_offset = sys.argv[2].lower() != 'no-offset' if len(sys.argv) > 2 else True
    
    test_complete_smart_chunking(pdf_name, apply_offset)
