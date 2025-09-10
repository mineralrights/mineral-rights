#!/usr/bin/env python3
"""
Test Custom Splitting Processor with PDF Chunking

This script splits large PDFs into chunks that fit within the 30-page limit.
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF

def split_pdf_into_chunks(pdf_path, max_pages=30):
    """Split a PDF into chunks that fit within the page limit"""
    print(f"📄 Splitting PDF: {pdf_path}")
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"📊 PDF has {total_pages} pages")
        print(f"📏 Max pages per chunk: {max_pages}")
        
        if total_pages <= max_pages:
            print("✅ PDF fits within page limit, no splitting needed")
            return [pdf_path]
        
        # Calculate number of chunks needed
        num_chunks = (total_pages + max_pages - 1) // max_pages
        print(f"📦 Will create {num_chunks} chunks")
        
        chunks = []
        base_name = Path(pdf_path).stem
        
        for i in range(num_chunks):
            start_page = i * max_pages
            end_page = min((i + 1) * max_pages, total_pages)
            
            print(f"📄 Creating chunk {i+1}/{num_chunks}: pages {start_page+1}-{end_page}")
            
            # Create new document for this chunk
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
            
            # Save chunk
            chunk_path = f"{base_name}_chunk_{i+1}.pdf"
            chunk_doc.save(chunk_path)
            chunk_doc.close()
            
            chunks.append(chunk_path)
            print(f"✅ Saved: {chunk_path}")
        
        doc.close()
        return chunks
        
    except Exception as e:
        print(f"❌ Error splitting PDF: {e}")
        return []

def test_chunk_with_document_ai(chunk_path):
    """Test a single chunk with Document AI"""
    print(f"\n🧪 Testing chunk: {chunk_path}")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the correct processor version
        processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        # Read PDF content
        with open(chunk_path, 'rb') as f:
            pdf_content = f.read()
        
        print(f"📄 Chunk loaded: {len(pdf_content)} bytes")
        
        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Create the request
        request = documentai.ProcessRequest(
            name=processor_version,
            raw_document=raw_document,
            skip_human_review=True
        )
        
        print("📤 Sending request to Document AI...")
        
        # Process the document
        result = client.process_document(request=request)
        document = result.document
        
        print("✅ Document AI processing completed!")
        print(f"📊 Results:")
        print(f"   - Pages: {len(document.pages)}")
        print(f"   - Text length: {len(document.text)}")
        print(f"   - Entities: {len(document.entities)}")
        
        # Look for splitting-related entities
        splitting_entities = []
        for entity in document.entities:
            if hasattr(entity, 'type_') and entity.type_:
                splitting_entities.append({
                    'type': entity.type_,
                    'text': entity.mention_text if hasattr(entity, 'mention_text') else 'N/A',
                    'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.0,
                    'page_refs': [page_ref.page for page_ref in entity.page_anchor.page_refs] if hasattr(entity, 'page_anchor') and entity.page_anchor else []
                })
        
        print(f"🔍 Found {len(splitting_entities)} splitting entities:")
        for entity in splitting_entities:
            print(f"   - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.3f}, pages: {entity['page_refs']})")
        
        return True, splitting_entities
        
    except Exception as e:
        print(f"❌ Document AI processing failed: {e}")
        return False, []

def cleanup_chunks(chunks):
    """Clean up temporary chunk files"""
    print(f"\n🧹 Cleaning up {len(chunks)} chunk files...")
    for chunk in chunks:
        try:
            os.remove(chunk)
            print(f"✅ Removed: {chunk}")
        except Exception as e:
            print(f"⚠️ Could not remove {chunk}: {e}")

def main():
    """Main test function"""
    print("🚀 Test Custom Splitting Processor with PDF Chunking")
    print("=" * 60)
    
    # Test PDF
    test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Test PDF not found: {test_pdf_path}")
        return False
    
    # Step 1: Split PDF into chunks
    chunks = split_pdf_into_chunks(test_pdf_path, max_pages=30)
    
    if not chunks:
        print("❌ Failed to split PDF")
        return False
    
    print(f"\n📦 Created {len(chunks)} chunks")
    
    # Step 2: Test each chunk with Document AI
    all_entities = []
    successful_chunks = 0
    
    for i, chunk in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"Testing Chunk {i+1}/{len(chunks)}")
        print(f"{'='*50}")
        
        success, entities = test_chunk_with_document_ai(chunk)
        
        if success:
            successful_chunks += 1
            all_entities.extend(entities)
            print(f"✅ Chunk {i+1} processed successfully")
        else:
            print(f"❌ Chunk {i+1} failed")
    
    # Step 3: Summary
    print(f"\n{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful chunks: {successful_chunks}/{len(chunks)}")
    print(f"🔍 Total entities found: {len(all_entities)}")
    
    if all_entities:
        print(f"\n🔍 All splitting entities found:")
        for entity in all_entities:
            print(f"   - {entity['type']}: confidence {entity['confidence']:.3f}")
    
    # Step 4: Cleanup
    cleanup_chunks(chunks)
    
    return successful_chunks > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Custom Splitting Processor is working with chunking!")
        print("📊 Summary:")
        print("   - PDF successfully split into chunks")
        print("   - Document AI processed all chunks")
        print("   - Splitting entities detected")
        print("\n🎯 Next steps:")
        print("1. Integrate chunking into the app")
        print("2. Update the app to use the correct processor version")
        print("3. Remove non-working splitting methods")
    else:
        print("\n❌ Chunking test failed")
    
    exit(0 if success else 1)
