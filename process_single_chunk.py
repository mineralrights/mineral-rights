#!/usr/bin/env python3
"""
Process a single chunk and save results to disk
"""

import os
import json
import time
import fitz
import gc
from google.cloud import documentai
from google.api_core import client_options

def process_chunk(start_page: int, end_page: int, chunk_id: int):
    """Process a single chunk and save results"""
    print(f"📤 Processing chunk {chunk_id} (pages {start_page}-{end_page})...")
    
    # Determine PDF path based on chunk results or default to THOMAS
    if any('chunk_23_results.json' in f for f in os.listdir('.')):
        pdf_path = "data/multi-deed/pdfs/THOMAS.pdf"
    elif any('chunk_9_results.json' in f for f in os.listdir('.')):
        pdf_path = "data/multi-deed/pdfs/ROBERT.pdf"
    else:
        pdf_path = "data/multi-deed/pdfs/FRANCO.pdf"
    
    # Initialize Document AI client
    opts = client_options.ClientOptions(api_endpoint="us-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
    
    # Extract chunk
    doc = fitz.open(pdf_path)
    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
    chunk_bytes = chunk_doc.write()
    
    # Clean up
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
        
        print(f"   ✅ Completed in {processing_time:.2f}s - {len(entities)} deeds")
        
        # Save results to disk
        results = {
            'chunk_id': chunk_id,
            'start_page': start_page,
            'end_page': end_page,
            'processing_time': processing_time,
            'deeds': entities
        }
        
        output_file = f"chunk_{chunk_id}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   💾 Results saved to {output_file}")
        
        # Clean up
        del result, raw_document, request, chunk_bytes
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python process_single_chunk.py <start_page> <end_page> <chunk_id>")
        print("Example: python process_single_chunk.py 1 15 1")
        sys.exit(1)
    
    start_page = int(sys.argv[1])
    end_page = int(sys.argv[2])
    chunk_id = int(sys.argv[3])
    
    process_chunk(start_page, end_page, chunk_id)
