#!/usr/bin/env python3
"""
Test Custom Splitting Processor with Imageless Mode

This script tests the custom splitting processor using imageless mode to increase the page limit.
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF

def test_with_imageless_mode():
    """Test with imageless mode to increase page limit"""
    print("🧪 Testing with imageless mode...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the correct processor version
        processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        if not os.path.exists(test_pdf_path):
            print(f"❌ Test PDF not found: {test_pdf_path}")
            return False
        
        # Read PDF content
        with open(test_pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        print(f"📄 PDF loaded: {len(pdf_content)} bytes")
        
        # Create the document with imageless mode
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Create the request with imageless mode
        request = documentai.ProcessRequest(
            name=processor_version,
            raw_document=raw_document,
            skip_human_review=True,
            # Enable imageless mode
            process_options=documentai.ProcessOptions(
                ocr_config=documentai.OcrConfig(
                    enable_imageless_mode=True
                )
            )
        )
        
        print("📤 Sending request with imageless mode...")
        
        # Process the document
        result = client.process_document(request=request)
        document = result.document
        
        print("✅ Document AI processing completed with imageless mode!")
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

def test_with_smaller_chunks():
    """Test with smaller chunks (15 pages max)"""
    print("\n🧪 Testing with smaller chunks (15 pages max)...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the correct processor version
        processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        # Split into smaller chunks
        doc = fitz.open(test_pdf_path)
        total_pages = len(doc)
        max_pages = 15  # Use 15 pages max
        
        print(f"📄 PDF has {total_pages} pages")
        print(f"📏 Max pages per chunk: {max_pages}")
        
        num_chunks = (total_pages + max_pages - 1) // max_pages
        print(f"📦 Will create {num_chunks} chunks")
        
        all_entities = []
        successful_chunks = 0
        
        for i in range(num_chunks):
            start_page = i * max_pages
            end_page = min((i + 1) * max_pages, total_pages)
            
            print(f"\n📄 Processing chunk {i+1}/{num_chunks}: pages {start_page+1}-{end_page}")
            
            # Create chunk
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
            
            # Save chunk temporarily
            chunk_path = f"temp_chunk_{i+1}.pdf"
            chunk_doc.save(chunk_path)
            chunk_doc.close()
            
            # Read chunk content
            with open(chunk_path, 'rb') as f:
                chunk_content = f.read()
            
            print(f"📄 Chunk loaded: {len(chunk_content)} bytes")
            
            # Create the document
            raw_document = documentai.RawDocument(
                content=chunk_content,
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
            
            print("✅ Chunk processed successfully!")
            print(f"📊 Results:")
            print(f"   - Pages: {len(document.pages)}")
            print(f"   - Text length: {len(document.text)}")
            print(f"   - Entities: {len(document.entities)}")
            
            # Look for splitting-related entities
            chunk_entities = []
            for entity in document.entities:
                if hasattr(entity, 'type_') and entity.type_:
                    chunk_entities.append({
                        'type': entity.type_,
                        'text': entity.mention_text if hasattr(entity, 'mention_text') else 'N/A',
                        'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.0,
                        'page_refs': [page_ref.page for page_ref in entity.page_anchor.page_refs] if hasattr(entity, 'page_anchor') and entity.page_anchor else [],
                        'chunk': i+1
                    })
            
            print(f"🔍 Found {len(chunk_entities)} entities in chunk {i+1}:")
            for entity in chunk_entities:
                print(f"   - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.3f}, pages: {entity['page_refs']})")
            
            all_entities.extend(chunk_entities)
            successful_chunks += 1
            
            # Clean up chunk file
            os.remove(chunk_path)
        
        doc.close()
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"✅ Successful chunks: {successful_chunks}/{num_chunks}")
        print(f"🔍 Total entities found: {len(all_entities)}")
        
        if all_entities:
            print(f"\n🔍 All splitting entities found:")
            for entity in all_entities:
                print(f"   - Chunk {entity['chunk']}: {entity['type']}: confidence {entity['confidence']:.3f}")
        
        return True, all_entities
        
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def main():
    """Main test function"""
    print("🚀 Test Custom Splitting Processor with Imageless Mode")
    print("=" * 60)
    
    # Step 1: Try imageless mode first
    print("🧪 Testing imageless mode...")
    success, entities = test_with_imageless_mode()
    
    if success:
        print("\n🎉 Imageless mode works!")
        print("📊 Summary:")
        print(f"   - Document processed successfully")
        print(f"   - Found {len(entities)} splitting entities")
        return True
    else:
        print("\n⚠️ Imageless mode failed, trying smaller chunks...")
        
        # Step 2: Try smaller chunks
        success, entities = test_with_smaller_chunks()
        
        if success:
            print("\n🎉 Smaller chunks work!")
            print("📊 Summary:")
            print(f"   - Document processed successfully")
            print(f"   - Found {len(entities)} splitting entities")
            return True
        else:
            print("\n❌ Both approaches failed")
            return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Custom Splitting Processor is working!")
        print("2. Update the app to use the correct processor version")
        print("3. Integrate chunking or imageless mode")
        print("4. Remove non-working splitting methods")
    else:
        print("\n🔧 Need to investigate further")
    
    exit(0 if success else 1)
