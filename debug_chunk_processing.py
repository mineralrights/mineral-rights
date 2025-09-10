#!/usr/bin/env python3
"""
Debug version of chunk processing to find the bug
"""

import os
import json
import time
import fitz
from google.cloud import documentai
from google.api_core import client_options

def debug_process_chunk(start_page: int, end_page: int, chunk_id: int):
    """Debug version of chunk processing"""
    print(f"📤 Debug processing chunk {chunk_id} (pages {start_page}-{end_page})...")
    
    pdf_path = "data/multi-deed/pdfs/THOMAS.pdf"
    
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
        
        print(f"   ✅ Document AI completed in {processing_time:.2f}s")
        print(f"   📊 Found {len(result.document.entities)} total entities")
        
        # Parse entities with debug info
        entities = []
        for i, entity in enumerate(result.document.entities):
            print(f"   Entity {i+1}: {entity.type_} (confidence={entity.confidence:.3f})")
            
            if entity.type_ == 'DEED':
                entity_dict = {
                    'type': entity.type_,
                    'confidence': entity.confidence,
                    'pages': [],
                    'chunk_id': chunk_id,
                    'chunk_start': start_page,
                    'chunk_end': end_page
                }
                
                # Extract page references with debug info
                if entity.page_anchor and entity.page_anchor.page_refs:
                    print(f"     Page anchor exists with {len(entity.page_anchor.page_refs)} refs")
                    for j, ref in enumerate(entity.page_anchor.page_refs):
                        print(f"       Ref {j+1}: page={ref.page}, confidence={ref.confidence}")
                        if ref.page is not None and str(ref.page).isdigit():
                            # Adjust page number to original PDF
                            adjusted_page = int(ref.page) + start_page
                            entity_dict['pages'].append(adjusted_page)
                            print(f"         Adjusted page: {adjusted_page}")
                        else:
                            print(f"         Skipped: page={ref.page}, isdigit={str(ref.page).isdigit() if ref.page is not None else 'None'}")
                else:
                    print(f"     No page anchor or page refs")
                
                print(f"     Final pages: {entity_dict['pages']}")
                
                # Only include deeds with valid pages
                if entity_dict['pages']:
                    entities.append(entity_dict)
                    print(f"     ✅ Added deed with {len(entity_dict['pages'])} pages")
                else:
                    print(f"     ❌ Skipped deed - no valid pages")
        
        print(f"   📊 Final result: {len(entities)} deeds")
        return entities, processing_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return [], time.time() - start_time

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python debug_chunk_processing.py <start_page> <end_page> <chunk_id>")
        sys.exit(1)
    
    start_page = int(sys.argv[1])
    end_page = int(sys.argv[2])
    chunk_id = int(sys.argv[3])
    
    debug_process_chunk(start_page, end_page, chunk_id)
