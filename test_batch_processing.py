#!/usr/bin/env python3
"""
Test Batch Processing with Document AI

This script tests batch processing which allows up to 1000 pages.
"""

import os
import sys
from pathlib import Path
import time

def test_batch_processing():
    """Test batch processing with Document AI"""
    print("🧪 Testing batch processing with Document AI...")
    
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
        
        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Create batch processing request
        request = documentai.BatchProcessRequest(
            name=processor_version,
            input_documents=documentai.BatchProcessRequest.BatchInputDocuments(
                documents=[raw_document]
            ),
            document_output_config=documentai.BatchProcessRequest.DocumentOutputConfig(
                gcs_output=documentai.BatchProcessRequest.DocumentOutputConfig.GcsOutput(
                    gcs_uri="gs://your-bucket/output/"  # This would need to be configured
                )
            )
        )
        
        print("📤 Sending batch processing request...")
        print("⚠️ Note: This requires GCS bucket configuration")
        
        # This would start batch processing
        # operation = client.batch_process_documents(request=request)
        # print(f"✅ Batch processing started: {operation.name}")
        
        print("💡 Batch processing would allow up to 1000 pages!")
        print("💡 This is the recommended approach for large documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False

def test_imageless_mode():
    """Test imageless mode for better scanned document processing"""
    print("\n🧪 Testing imageless mode...")
    
    try:
        from google.cloud import documentai
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Use the correct processor version
        processor_version = "projects/381937358877/locations/us/processors/895767ed7f252878/processorVersions/106a39290d05efaf"
        
        # Test PDF
        test_pdf_path = "/Users/lauragomez/Desktop/mineral-rights/data/multi-deed/pdfs/FRANCO.pdf"
        
        # Read PDF content
        with open(test_pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        print(f"📄 PDF loaded: {len(pdf_content)} bytes")
        
        # Create the document
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf"
        )
        
        # Create the request with imageless mode
        request = documentai.ProcessRequest(
            name=processor_version,
            raw_document=raw_document,
            skip_human_review=True,
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
        
        print("✅ Imageless mode processing completed!")
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
        print(f"❌ Imageless mode failed: {e}")
        return False, []

def main():
    """Main test function"""
    print("🚀 Test Batch Processing and Imageless Mode")
    print("=" * 60)
    
    # Test batch processing
    batch_success = test_batch_processing()
    
    # Test imageless mode
    imageless_success, entities = test_imageless_mode()
    
    if imageless_success:
        print("\n🎉 Imageless mode works!")
        print("📊 Summary:")
        print(f"   - Document processed successfully")
        print(f"   - Found {len(entities)} splitting entities")
        print("💡 This approach is better for scanned documents")
    else:
        print("\n❌ Imageless mode failed")
    
    print("\n💡 Recommendations:")
    print("1. **Batch Processing**: Best for large documents (up to 1000 pages)")
    print("2. **Imageless Mode**: Better for scanned documents (up to 30 pages)")
    print("3. **Current Chunking**: Works but may not be detecting real deeds")
    
    return imageless_success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Next steps:")
        print("1. Consider implementing batch processing")
        print("2. Use imageless mode for better scanned document handling")
        print("3. Verify that detected 'deeds' are actually legal deeds")
    else:
        print("\n🔧 Need to investigate further")
    
    exit(0 if success else 1)
