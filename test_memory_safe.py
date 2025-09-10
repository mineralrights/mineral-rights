#!/usr/bin/env python3
"""
Memory-safe chunking test - process just 2 chunks to avoid memory issues
"""

import os
import json
import time
import fitz
import gc
from google.cloud import documentai
from google.api_core import client_options

def process_single_chunk(pdf_path: str, start_page: int, end_page: int, chunk_id: int, client, processor_name: str):
    """Process a single chunk with aggressive memory cleanup"""
    print(f"📤 Processing chunk {chunk_id} (pages {start_page}-{end_page})...")
    
    # Extract chunk
    doc = fitz.open(pdf_path)
    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
    chunk_bytes = chunk_doc.write()
    
    # Aggressive cleanup
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
            if entity.type_ == 'DEED':
                entity_dict = {
                    'type': entity.type_,
                    'confidence': entity.confidence,
                    'pages': [],
                    'chunk_id': chunk_id
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
        for i, deed in enumerate(entities):
            print(f"      Deed {i+1}: Pages {deed['pages']} (Confidence: {deed['confidence']:.3f})")
        
        # Clean up result object
        del result, raw_document, request, chunk_bytes
        gc.collect()
        
        return entities, processing_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return [], time.time() - start_time

def test_memory_safe():
    """Test memory-safe chunking with just 2 chunks"""
    print("🧪 Memory-Safe Chunking Test - First 2 chunks of FRANCO.pdf")
    print("=" * 50)
    
    pdf_path = "data/multi-deed/pdfs/FRANCO.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Initialize Document AI client
    opts = client_options.ClientOptions(api_endpoint="us-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
    
    # Define just 2 chunks
    chunks = [
        (1, 15),   # Pages 1-15
        (13, 27)   # Pages 13-27 (3-page overlap)
    ]
    
    print(f"📄 Processing {len(chunks)} chunks:")
    for i, (start, end) in enumerate(chunks):
        print(f"   Chunk {i+1}: Pages {start}-{end} ({end-start+1} pages)")
    
    # Process chunks one by one with cleanup
    all_deeds = []
    total_time = 0
    
    for i, (start_page, end_page) in enumerate(chunks):
        print(f"\n🔄 Processing chunk {i+1}/{len(chunks)}...")
        
        deeds, processing_time = process_single_chunk(
            pdf_path, start_page, end_page, i+1, client, processor_name
        )
        
        all_deeds.extend(deeds)
        total_time += processing_time
        
        # Force garbage collection after each chunk
        gc.collect()
        print(f"   🧹 Memory cleaned up after chunk {i+1}")
    
    # Simple merge logic
    print(f"\n📊 Merging Results:")
    print(f"   - Total deeds found: {len(all_deeds)}")
    
    # Sort by starting page
    all_deeds.sort(key=lambda x: min(x['pages']) if x['pages'] else 0)
    
    merged_deeds = []
    for deed in all_deeds:
        if not deed['pages']:
            continue
            
        deed_start = min(deed['pages'])
        deed_end = max(deed['pages'])
        
        # Check if this deed should be merged with the last merged deed
        if merged_deeds:
            last_deed = merged_deeds[-1]
            last_start = min(last_deed['pages'])
            last_end = max(last_deed['pages'])
            
            # If overlap (at least 2 pages), merge
            overlap_start = max(deed_start, last_start)
            overlap_end = min(deed_end, last_end)
            overlap_pages = max(0, overlap_end - overlap_start + 1)
            
            if overlap_pages >= 2:
                # Merge pages
                merged_pages = sorted(list(set(last_deed['pages'] + deed['pages'])))
                merged_deeds[-1] = {
                    'type': 'DEED',
                    'confidence': max(last_deed['confidence'], deed['confidence']),
                    'pages': merged_pages,
                    'chunk_id': f"{last_deed.get('chunk_id', 'unknown')}+{deed.get('chunk_id', 'unknown')}"
                }
                print(f"   Merged deed: Pages {merged_pages} (overlap: {overlap_pages} pages)")
            else:
                merged_deeds.append(deed)
                print(f"   New deed: Pages {deed['pages']}")
        else:
            merged_deeds.append(deed)
            print(f"   First deed: Pages {deed['pages']}")
    
    print(f"\n📊 Final Results:")
    print(f"   - Chunks processed: {len(chunks)}")
    print(f"   - Total processing time: {total_time:.2f}s")
    print(f"   - Deeds before merge: {len(all_deeds)}")
    print(f"   - Deeds after merge: {len(merged_deeds)}")
    
    # Show final merged deeds
    print(f"\n📄 Final Merged Deeds:")
    for i, deed in enumerate(merged_deeds):
        print(f"   Deed {i+1}: Pages {deed['pages']} (Confidence: {deed['confidence']:.3f})")
    
    # Load ground truth for comparison
    with open("data/multi-deed/normalized_boundaries/FRANCO.normalized.json", 'r') as f:
        gt = json.load(f)
    
    gt_deed_count = gt['deed_count']
    gt_page_starts = gt['page_starts']
    
    print(f"\n📈 Comparison with Ground Truth:")
    print(f"   - Ground truth deeds: {gt_deed_count}")
    print(f"   - Detected deeds: {len(merged_deeds)}")
    print(f"   - Detection ratio: {len(merged_deeds)/gt_deed_count:.2f}x")
    
    # Check for systematic offset
    if len(merged_deeds) >= 3 and len(gt_page_starts) >= 3:
        ai_starts = [min(d['pages']) for d in merged_deeds[:3]]
        gt_starts = gt_page_starts[:3]
        offsets = [ai - gt for ai, gt in zip(ai_starts, gt_starts)]
        print(f"   - First 3 offsets: {offsets}")
        if len(set(offsets)) == 1:
            print(f"   - Systematic offset: {offsets[0]} pages")
        else:
            print(f"   - No systematic offset detected")

if __name__ == "__main__":
    test_memory_safe()
