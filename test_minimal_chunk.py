#!/usr/bin/env python3
"""
Minimal test - just process first 25 pages of FRANCO.pdf
"""

import os
import fitz
import time
from google.cloud import documentai
from google.api_core import client_options

def test_minimal_chunk():
    """Test processing just the first 25 pages"""
    print("🧪 Minimal Chunk Test - First 25 pages of FRANCO.pdf")
    print("=" * 50)
    
    pdf_path = "data/multi-deed/pdfs/FRANCO.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Initialize Document AI client
    opts = client_options.ClientOptions(api_endpoint="us-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    processor_name = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
    
    # Extract first 15 pages (within non-imageless limit)
    print("📄 Extracting first 15 pages...")
    doc = fitz.open(pdf_path)
    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=0, to_page=14)  # Pages 0-14 (first 15 pages)
    chunk_bytes = chunk_doc.write()
    
    print(f"📦 Chunk size: {len(chunk_bytes)} bytes")
    
    # Clean up
    doc.close()
    chunk_doc.close()
    
    # Process chunk
    print("📤 Processing chunk...")
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
                        entity_dict['page_refs'].append({
                            'page': str(ref.page),
                            'confidence': ref.confidence
                        })
            
            entities.append(entity_dict)
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 Found {len(entities)} entities:")
        
        deeds = [e for e in entities if e['type'] == 'DEED']
        covers = [e for e in entities if e['type'] == 'COVER']
        
        print(f"   - DEED entities: {len(deeds)}")
        print(f"   - COVER entities: {len(covers)}")
        
        # Show deed details
        for i, deed in enumerate(deeds):
            pages = [ref['page'] for ref in deed['page_refs']]
            print(f"   Deed {i+1}: Pages {pages} (Confidence: {deed['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_minimal_chunk()
